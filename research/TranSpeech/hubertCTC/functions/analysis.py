import math

import numpy as np
import torch
import transformers

from models.ecapa import ECAPA_TDNN
from models.yin import *


class Linguistic(torch.nn.Module):
    def __init__(self, conf=None):
        """we decided to use the intermediate features of XLSR-53. More specifically, we used the output from the 12th layer of the 24-layer transformer encoder.

        Args:
            conf:
        """
        super(Linguistic, self).__init__()
        self.conf = conf

        self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            param.grad = None
        self.wav2vec2.eval()

    def forward(self, x):
        """

        Args:
            x: torch.Tensor of shape (B x t)

        Returns:
            y: torch.Tensor of shape(B x C x t)
        """
        with torch.no_grad():
            outputs = self.wav2vec2(x, output_hidden_states=True)
        y = outputs.hidden_states[12]  # B x t x C(1024)
        y = y.permute((0, 2, 1))  # B x t x C -> B x C x t
        return y

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        # for module in self.children():
        #     module.train(mode)
        return self


class Speaker(torch.nn.Module):
    def __init__(self, conf=None):
        """We train a speaker embedding network that uses the 1st layer of XLSR-53 as an input. For the speaker embedding network, we borrow the neural architecture from a state-of-the-art speaker recognition network [14]

        Args:
            conf:
        """
        super(Speaker, self).__init__()
        self.conf = conf

        self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            param.grad = None
        self.wav2vec2.eval()

        # c_in = 1024 for wav2vec2
        # original paper[14] used 512 and 192 for c_mid and c_out, respectively
        self.spk = ECAPA_TDNN(c_in=1024, c_mid=512, c_out=192)

    def forward(self, x):
        """

        Args:
            x: torch.Tensor of shape (B x t)

        Returns:
            y: torch.Tensor of shape (B x 192)
        """
        with torch.no_grad():
            outputs = self.wav2vec2(x, output_hidden_states=True)
        y = outputs.hidden_states[1]  # B x t x C(1024)
        y = y.permute((0, 2, 1))  # B x t x C -> B x C x t
        y = self.spk(y)  # B x C(1024) x t -> B x D(192)
        y = torch.nn.functional.normalize(y, p=2, dim=-1)
        return y

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        # for module in self.children():
        #     module.train(mode)
        self.spk.train(mode)
        return self


class Energy(torch.nn.Module):
    def __init__(self, conf=None):
        super(Energy, self).__init__()
        self.conf = conf

    def forward(self, mel):
        """For the energy feature, we simply took an average from a log-mel spectrogram along the frequency axis.

        Args:
            mel: torch.Tensor of shape (B x t x C)

        Returns:
            y: torch.Tensor of shape (B x 1 x C)
        """
        y = torch.mean(mel, dim=1, keepdim=True)  # B x 1(channel) x t
        return y


class Pitch(torch.nn.Module):
    def __init__(self, conf=None):
        super(Pitch, self).__init__()
        self.conf = conf

    @staticmethod
    def midi_to_lag(m: int, sr: int, semitone_range: float = 12):
        """converts midi-to-lag, eq. (4)

        Args:
            m: midi
            sr: sample_rate
            semitone_range:

        Returns:
            lag: time lag(tau, c(m)) calculated from midi, eq. (4)

        """
        f = 440 * math.pow(2, (m - 69) / semitone_range)
        lag = sr / f
        return lag

    @staticmethod
    def yingram_from_cmndf(cmndfs: torch.Tensor, ms: list, sr: int = 22050) -> torch.Tensor:
        """ yingram calculator from cMNDFs(cumulative Mean Normalized Difference Functions)

        Args:
            cmndfs: torch.Tensor
                calculated cumulative mean normalized difference function
                for details, see models/yin.py or eq. (1) and (2)
            ms: list of midi(int)
            sr: sampling rate

        Returns:
            y:
                calculated batch yingram


        """
        c_ms = np.asarray([Pitch.midi_to_lag(m, sr) for m in ms])
        c_ms = torch.from_numpy(c_ms).to(cmndfs.device)
        c_ms_ceil = torch.ceil(c_ms).long().to(cmndfs.device)
        c_ms_floor = torch.floor(c_ms).long().to(cmndfs.device)

        y = (cmndfs[:, c_ms_ceil] - cmndfs[:, c_ms_floor]) / (c_ms_ceil - c_ms_floor).unsqueeze(0) * (
                c_ms - c_ms_floor).unsqueeze(0) + cmndfs[:, c_ms_floor]
        return y

    @staticmethod
    def yingram(x: torch.Tensor, W: int = 2048, tau_max: int = 2048, sr: int = 22050, w_step: int = 256):
        """calculates yingram from raw audio (multi segment)

        Args:
            x: raw audio, torch.Tensor of shape (t)
            W: yingram Window Size
            tau_max:
            sr: sampling rate
            w_step: yingram bin step size

        Returns:
            yingram: yingram. torch.Tensor of shape (80 x t')

        """
        # x.shape: t
        w_len = W

        startFrames = list(range(0, x.shape[-1] - w_len, w_step))
        startFrames = np.asarray(startFrames)
        # times = startFrames / sr
        frames = [x[..., t:t + W] for t in startFrames]
        frames_torch = torch.stack(frames, dim=0).to(x.device)

        # If not using gpu, or torch not compatible, implemented numpy batch function is still fine
        dfs = differenceFunctionTorch(frames_torch, frames_torch.shape[-1], tau_max)
        cmndfs = cumulativeMeanNormalizedDifferenceFunctionTorch(dfs, tau_max)

        midis = list(range(5, 85))
        yingram = Pitch.yingram_from_cmndf(cmndfs, midis, sr)
        return yingram

    @staticmethod
    def yingram_batch(x: torch.Tensor, W: int = 2048, tau_max: int = 2048, sr: int = 22050, w_step: int = 256):
        """calculates yingram from batch raw audio.
        currently calculates batch-wise through for loop, but seems it can be implemented to act batch-wise

        Args:
            x: torch.Tensor of shape (B x t)
            W:
            tau_max:
            sr:
            w_step:

        Returns:
            yingram: yingram. torch.Tensor of shape (B x 80 x t')

        """
        batch_results = []
        for i in range(len(x)):
            yingram = Pitch.yingram(x[i], W, tau_max, sr, w_step)
            batch_results.append(yingram)
        result = torch.stack(batch_results, dim=0).float()
        result = result.permute((0, 2, 1)).to(x.device)
        return result


class Analysis(torch.nn.Module):
    def __init__(self, conf=None):
        """joins all analysis modules into one

        Args:
            conf:
        """
        super(Analysis, self).__init__()
        self.conf = conf

        self.linguistic = Linguistic()
        self.speaker = Speaker()
        self.energy = Energy()
        self.pitch = Pitch()


if __name__ == '__main__':
    import torch
    from datasets.custom import CustomDataset
    from omegaconf import OmegaConf, DictConfig
    from datasets.functional import f, g, get_pitch_median

    conf_audio = OmegaConf.load('configs/audio/22k.yaml')
    conf = DictConfig({'audio': conf_audio})
    dataset = CustomDataset(conf)
    wav = torch.randn(2, 34816) # 272


    linguistic = Linguistic()
    speaker = Speaker()
    energy = Energy()
    pitch = Pitch()

    with torch.no_grad():
        mel = dataset.load_mel_from_audio(wav, conf.audio)
        print(mel.shape)

        ps = pitch.yingram_batch(wav)
        print(ps.shape)


        lps = linguistic(wav)
        print(lps.shape)

        s = speaker(wav)
        print(s.shape)

        e = energy(mel)
        print(e.shape)

