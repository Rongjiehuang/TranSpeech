from fairseq.data.audio.speech_to_speech_dataset import *
from fairseq.data.audio.speech_to_text_dataset import (
    _collate_frames,
)

class SpeechToSpeechFastTranslateDatasetCreator(SpeechToSpeechDatasetCreator):
    # mandatory columns
    KEY_ID, KEY_SRC_AUDIO, KEY_SRC_N_FRAMES = "id", "src_audio", "src_n_frames"
    KEY_TGT_AUDIO, KEY_TGT_N_FRAMES = "tgt_audio", "tgt_n_frames"

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        data_cfg: S2SDataConfig,
        target_is_code: bool = False,
        target_dictionary: Dictionary = None,
        n_frames_per_step: int = 1,
        multitask: Optional[Dict] = None,
    ) -> SpeechToSpeechDataset:
        audio_root = Path(data_cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        src_audio_paths = [
            (audio_root / s[cls.KEY_SRC_AUDIO]).as_posix() for s in samples
        ]
        tgt_audio_paths = [
            s[cls.KEY_TGT_AUDIO]
            if target_is_code
            else (audio_root / s[cls.KEY_TGT_AUDIO]).as_posix()
            for s in samples
        ]
        src_n_frames = [int(s[cls.KEY_SRC_N_FRAMES]) for s in samples]
        tgt_n_frames = [int(s[cls.KEY_TGT_N_FRAMES]) for s in samples]

        has_multitask = len(multitask) > 0
        dataset_cls = (
            SpeechToSpeechFastTranslateMultitaskDataset if has_multitask else SpeechToSpeechFastTranslateDataset
        )

        ds = dataset_cls(
            split_name,
            is_train_split,
            data_cfg,
            src_audio_paths,
            src_n_frames,
            tgt_audio_paths,
            tgt_n_frames,
            ids,
            target_is_code,
            target_dictionary,
            n_frames_per_step,
        )

        if has_multitask:
            for task_name, task_obj in multitask.items():
                task_data = TextTargetMultitaskData(
                    task_obj.args, split_name, task_obj.target_dictionary
                )
                ds.add_multitask_dataset(task_name, task_data)
        return ds


class SpeechToSpeechFastTranslateDataset(SpeechToSpeechDataset):

    def collater(
        self, samples: List[SpeechToSpeechDatasetItem], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        frames = _collate_frames([x.source for x in samples], self.cfg.use_audio_input)
        # sort samples by descending number of frames
        n_frames = torch.tensor([x.source.size(0) for x in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, prev_output_tokens, target_lengths = self._collate_target(samples)
        target = target.index_select(0, order)
        target_lengths = target_lengths.index_select(0, order)
        prev_output_tokens = prev_output_tokens.index_select(0, order)
        ntokens = sum(x.target.size(0) for x in samples)

        tgt_speakers = None
        if self.cfg.target_speaker_embed:
            tgt_speakers = _collate_frames(
                [x.target_speaker for x in samples], is_audio_input=True
            ).index_select(0, order)

        net_input = {
            "src_tokens": frames,
            "src_lengths": n_frames,
            "target": target,
            "target_lengths": target_lengths,
            "prev_output_tokens": prev_output_tokens,
            "tgt_speaker": tgt_speakers,  # TODO: unify "speaker" and "tgt_speaker"
        }
        out = {
            "id": indices,
            "net_input": net_input,
            "speaker": tgt_speakers,  # to support Tacotron2 loss for speech-to-spectrogram model
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        if return_order:
            out["order"] = order
        return out


class SpeechToSpeechFastTranslateMultitaskDataset(SpeechToSpeechFastTranslateDataset):
    def __init__(self, *argv):
        super().__init__(*argv)
        self.multitask_data = {}

    def add_multitask_dataset(self, task_name, task_data):
        self.multitask_data[task_name] = task_data

    def __getitem__(
        self, index: int
    ) -> Tuple[SpeechToSpeechDatasetItem, Dict[str, torch.Tensor]]:
        s2s_data = super().__getitem__(index)

        multitask_target = {}
        sample_id = self.ids[index]
        for task_name, task_dataset in self.multitask_data.items():
            multitask_target[task_name] = task_dataset.get(sample_id)

        return s2s_data, multitask_target

    def collater(
        self, samples: List[Tuple[SpeechToSpeechDatasetItem, Dict[str, torch.Tensor]]]
    ) -> Dict:
        if len(samples) == 0:
            return {}

        out = super().collater([s for s, _ in samples], return_order=True)
        order = out["order"]
        del out["order"]

        for task_name, task_dataset in self.multitask_data.items():
            if "multitask" not in out:
                out["multitask"] = {}
            d = [s[task_name] for _, s in samples]
            task_target = task_dataset.collater(d)
            out["multitask"][task_name] = {
                "target": task_target["target"].index_select(0, order),
                "target_lengths": task_target["target_lengths"].index_select(0, order),
                "ntokens": task_target["ntokens"],
            }
            out["multitask"][task_name]["net_input"] = {
                "prev_output_tokens": task_target["prev_output_tokens"].index_select(
                    0, order
                ),
            }

        return out