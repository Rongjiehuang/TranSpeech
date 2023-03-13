import pandas as pd
from tqdm import tqdm
from examples.speech_to_text.data_utils import save_df_to_tsv
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, help='input wave directory for information enhancement')
    parser.add_argument('--txt', type=str, help='input text')
    parser.add_argument('--unit', type=str, help='output units directory')

    args = parser.parse_args()


    audio = {}
    for split in ['train', 'valid']:
        tsv_file = f'{args.manifest}/{split}.tsv'
        oldunit_file = f'{args.txt}/{split}.txt'
        write_file = f'{args.unit}/{split}.unit'

        with open(oldunit_file, 'r', encoding='utf-8') as f_oldunit_file:
            for i, line in tqdm(enumerate(f_oldunit_file)):
                line = line.strip('\n').split('|')
                id = line[0]
                txt = line[1].split(' ')
                out = [u for i, u in enumerate(txt) if i == 0 or u != txt[i - 1]]  # 去除重复unit
                audio[id] = ' '.join(out)

        f_write = open(write_file, mode='w')
        with open(tsv_file, 'r', encoding='utf-8') as f_tsv_file:
            for i, line in tqdm(enumerate(f_tsv_file)):
                if i == 0: continue
                line = line.strip('\n').split('\t')
                id = line[0].replace('.wav', '')
                unit = audio[id]
                f_write.write(f"{unit}\n")