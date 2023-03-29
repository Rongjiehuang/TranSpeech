import os
import argparse
from research.TranSpeech.hubertCTC.functions.functional import *
from tqdm import tqdm
from pathlib import Path
from scipy.io import wavfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav', type=str, help='input wave directory for information enhancement')
    parser.add_argument('--out', type=str, help='output directory')

    args = parser.parse_args()

    splits = ['train', 'test', 'dev']
    for split in splits:
        wavpaths = list((Path(f'{args.wav}/{split}/')).glob('*.wav'))
        savepath = f'{args.out}/{split}'
        pitch = []
        energy = []
        sr = 16000
        # Get Pitch Median
        for wavpath in tqdm(wavpaths):
            name = wavpath.stem
            wav_numpy, sr = librosa.core.load(wavpath, sr=sr)
            wav_torch = torch.from_numpy(wav_numpy).float()
            pitch_temp = get_median(wav_torch, sr)
            if pitch_temp < 250:
                pitch += [pitch_temp]

        pitch_mean = np.mean(pitch)
        os.makedirs(savepath, exist_ok=True)
        os.makedirs(f'{savepath}/temp', exist_ok=True)

        # Norm Pitch and Get Energy Median
        for wavpath in tqdm(wavpaths):
            name = wavpath.stem
            wav_numpy, sr = librosa.core.load(wavpath, sr=sr)
            wav_torch = torch.from_numpy(wav_numpy).float()
            wav_16k_torch_f = manipulate_median(wav_torch, sr, pitch_mean)
            wavfile.write(f'{savepath}/temp/{name}_p.wav', sr, wav_16k_torch_f.numpy())
            energy += [np.mean(wav_16k_torch_f.numpy().abs())]

        os.makedirs(f'{savepath}/result', exist_ok=True)
        energy_mean = np.mean(energy)
        # Norm Energy
        for wavpath in tqdm(wavpaths):
            name = wavpath.stem
            wav_numpy, sr = librosa.core.load(f'{savepath}/temp/{name}_p.wav', sr=sr)
            wav_numpy = wav_numpy / np.mean(wav_numpy.abs()) * energy_mean
            wavfile.write(f'{savepath}/result/{name}.wav', sr, wav_numpy)
        os.system(f'rm -r {savepath}/temp')