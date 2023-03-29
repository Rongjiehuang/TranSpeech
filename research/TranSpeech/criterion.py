# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch import Tensor
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.ctc import CtcCriterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
    label_smoothed_nll_loss,
)
# from fairseq.criterions.speech_to_speech_criterion import MultitaskCriterion

import torch.nn.functional as F


class MultitaskCriterion:
    def __init__(self, multitask_tasks):
        self.multitask_criterion = {}
        self.multitask_loss_weight = {}
        for task_name, task_obj in multitask_tasks.items():
            if task_obj.args.decoder_type == "ctc":
                self.multitask_criterion[task_name] = CtcCriterion(
                    task_obj.args.criterion_cfg, task_obj
                )
            else:
                self.multitask_criterion[
                    task_name
                ] = LabelSmoothedCrossEntropyCriterion(
                    task_obj,
                    task_obj.args.criterion_cfg.sentence_avg,
                    label_smoothing=task_obj.args.criterion_cfg.label_smoothing,
                )

    def set_multitask_loss_weight(self, task_name, weight=0.0):
        self.multitask_loss_weight[task_name] = weight

    def get_multitask_loss(self, model, sample, model_out):
        logging_output = {}
        loss = 0.0
        for task_name, task_criterion in self.multitask_criterion.items():
            layer_id = task_criterion.task.args.input_layer

            if isinstance(task_criterion, CtcCriterion):
                if task_criterion.task.args.input_from == "encoder":
                    task_sample = {   # encoder: ctc
                        "net_input": {
                            "src_tokens": model_out["encoder_states"][
                                layer_id
                            ],  # check batch idx
                            "src_lengths": sample["net_input"]["src_lengths"] // 4,
                        },
                        "id": sample["id"],
                    }
                else:
                    task_sample = {     # decoder: ctc / asr
                        "net_input": {
                            "src_tokens": model_out["inner_states"][layer_id],
                            "src_lengths": sample["net_input"]["target_lengths"],
                        },
                        "id": sample["id"],
                    }
            else:
                task_sample = {
                    "net_input": {
                        "src_tokens": sample["multitask"][task_name]["net_input"][
                            "prev_output_tokens"
                        ],
                        "encoder_out": {
                            "encoder_out": [model_out["encoder_states"][layer_id]],
                            "encoder_padding_mask": model_out["encoder_padding_mask"],
                        },
                    }
                }


            for key in ["target", "ntokens"]: # "target_lengths",
                task_sample[key] = sample["multitask"][task_name][key]  # sample[key]

            task_loss, task_sample_size, task_logging_output = task_criterion(
                model.multitask_decoders[task_name], task_sample
            )

            loss = loss + self.multitask_loss_weight[task_name] * task_loss
            task_logging_output["loss_weight"] = self.multitask_loss_weight[task_name]
            logging_output[task_name] = task_logging_output
        return loss, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        for task_name in logging_outputs[0]["multitask"].keys():
            # different criterion may return different logging
            # currently only reduce on loss, the most common one
            # ideally the way that losses are reduced should also depend on the task type
            loss_sum = sum(
                log["multitask"][task_name].get("loss", 0) for log in logging_outputs
            )
            sample_size = sum(
                log["multitask"][task_name].get("sample_size", 0)
                for log in logging_outputs
            )

            metrics.log_scalar(
                f"multitask_{task_name}_loss",
                loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )

            loss_weight = logging_outputs[0]["multitask"][task_name].get(
                "loss_weight", 0
            )
            metrics.log_scalar(
                f"multitask_{task_name}_loss_weight",
                loss_weight,
                weight=0,
                priority=250,
            )


@register_criterion(
    "nar_speech_to_unit", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class NARSpeechToUnitMultitaskTaskCriterion(
    LabelSmoothedCrossEntropyCriterion, MultitaskCriterion
):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )
        MultitaskCriterion.__init__(self, task.multitask_tasks)

    def get_lprobs_and_target(self, model, output, sample):
        if len(output) == 2:
            masks = output[1]
            net_output = output[0]
        else:
            masks = None
            net_output = output

        target = model.get_targets(sample, net_output) # B, T
        if masks is not None:
            net_output, target = [net_output[masks]], target[masks]
        lprobs = model.get_normalized_probs(net_output, log_probs=True) # B, T, n_class

        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)


    def forward(self, model, sample, reduce=True):
        '''
        model: S2STransformerEncoder + TransformerUnitDecoder
        sample: ['id', 'net_input', 'speaker', 'target', 'target_lengths', 'ntokens', 'nsentences']
        sample['net_input']: ['src_tokens', 'src_lengths', 'target', 'target_lengths', 'prev_output_tokens', 'tgt_speaker']

        net_output: [B, L1, dictionary size]
        extra: ['attn', 'inner_states', 'encoder_states', 'encoder_padding_mask']
        '''
        # 1: pad; 3: mask

        net_output, extra = model(
            src_tokens=sample["net_input"]["src_tokens"],
            src_lengths=sample["net_input"]["src_lengths"],
            prev_output_tokens=sample["net_input"]["prev_target"],
            tgt_tokens=sample["net_input"]["target"],
            tgt_speaker=sample["net_input"]["tgt_speaker"],
            return_all_hiddens=True,
        )

        loss, nll_loss = self.compute_loss(model, [net_output, extra['word_ins_mask']], sample, reduce=reduce)
        loss_length, nll_loss_length = self.compute_loss(model, [extra['length_out']], {'target': extra['length_tgt']}, reduce=reduce) # "loss", "nll_loss", "factor"

        loss += loss_length
        nll_loss += nll_loss_length

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "loss_length": loss_length.data,
            "nll_loss_length": nll_loss_length.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        n_correct, total = self.compute_accuracy(model, [net_output], sample)
        logging_output["n_correct"] = utils.item(n_correct.data)
        logging_output["total"] = utils.item(total.data)


        if len(self.multitask_criterion) == 0:
            return loss, sample_size, logging_output

        # multitask
        multitask_loss, multitask_log = self.get_multitask_loss(model, sample, extra)

        loss += multitask_loss
        logging_output["multitask"] = multitask_log

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        loss_length_sum = sum(log.get("loss_length", 0) for log in logging_outputs)
        nll_loss_length_sum = sum(log.get("nll_loss_length", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)


        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "loss_length", loss_length_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss_length", nll_loss_length_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )


        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

        # inference metrics
        if "targ_frames" in logging_outputs[0]:
            n = sum(log.get("norm_frames", 0) for log in logging_outputs)
            for key, new_key in [
                ("mcd_loss", "mcd_loss"),
                ("pred_frames", "pred_ratio"),
                ("nins", "ins_rate"),
                ("ndel", "del_rate"),
            ]:
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(new_key, val / n, n, round=3)

        if "multitask" not in logging_outputs[0]:
            return

        MultitaskCriterion.reduce_metrics(logging_outputs)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False

