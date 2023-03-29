from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from research.TranSpeech.hifigan.env import AttrDict
from research.TranSpeech.hifigan.meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from research.TranSpeech.hifigan.models import Generator
from datetime import datetime
device = None
import numpy as np
from librosa.util import normalize

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict



def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    rtfs = []
    with torch.no_grad():
        for i, filname in enumerate(filelist):

            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            x = get_mel(wav.unsqueeze(0))

            start = datetime.now()
            y_g_hat = generator(x)
            end = datetime.now()
            inference_time = (end - start).total_seconds()


            audio = y_g_hat.squeeze()
            rtf = compute_rtf(audio, inference_time, sample_rate=22050)
            print('rtf:', rtf)

            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            rtfs.append(rtf)

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)
    print(f'Done. RTF estimate: {np.mean(rtfs)} ± {np.std(rtfs)}')

def compute_rtf(sample, generation_time, sample_rate=22050):
    """
    Computes RTF for a given sample.
    """
    total_length = sample.shape[-1]
    return float(generation_time * sample_rate / total_length)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

