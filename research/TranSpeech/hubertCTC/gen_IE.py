import random
from research.TranSpeech.hubertCTC.Resample import InterpLnr
from research.TranSpeech.hifigan.meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from research.TranSpeech.hifigan.models import Generator
import torch
from pathlib import Path
import os
from research.TranSpeech.hifigan.env import AttrDict
import json
from tqdm import tqdm
from scipy.io.wavfile import write
from research.TranSpeech.hifigan.inference import load_checkpoint
global h
from research.TranSpeech.hubertCTC.functions.functional import f as func
from research.TranSpeech.tfcompat.hparam import HParams
import argparse

def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


hparams = HParams(
    # interp
    min_len_seg=19,
    max_len_seg=32,
    min_len_seq=300,
    max_len_seq=800,
    max_len_pad=800,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--ckpt', type=str, help='hifigan checkpoint')
    parser.add_argument('--wav', type=str, help='input wave directory for information enhancement')
    parser.add_argument('--out', type=str, help='output directory')

    args = parser.parse_args()

    config_file = os.path.join(os.path.split(args.ckpt)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    Interp = InterpLnr(hparams).to(device)
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(args.ckpt, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    with torch.no_grad():
        wavfiles = list((Path(args.wav)).glob('*.wav'))
        for wavfile in tqdm(wavfiles):
            filename = wavfile.stem
            wav, sr = load_wav(wavfile)
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            x = get_mel(wav.unsqueeze(0))  # [1, 80, T]

            # Rhythm
            seq = Interp(x.transpose(1, 2), torch.tensor(x.shape[-1]).to(device))[0].transpose(0, 1).unsqueeze(0)
            y_g_hat = generator(seq)

            # Pitch
            audio = func(y_g_hat.squeeze().cpu(), sr=16000)

            # Energy
            audio = audio * random.random()

            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            output_file = os.path.join(f'{args.out}/{filename}.wav')
            write(output_file, h.sampling_rate, audio)










