import pandas as pd
from tqdm import tqdm
from examples.speech_to_text.data_utils import save_df_to_tsv



audio = {}
for split in ['train', 'valid']:
    read_file = f'/home/huangrongjie/Project/Translation/speechresynthesis/data/LJSpeech_HuBERT_Base_KM100_norm_libriTTS_train_100_RR/{split}/hypo.units'
    oldunit_file = f'/home/huangrongjie/Project/Translation/speechresynthesis/data/LJSpeech_HuBERT_Base_KM100_tuned/{split}.txt'
    write_file = f'/home/huangrongjie/Project/Translation/speechresynthesis/data/LJSpeech_HuBERT_Base_KM100_norm_libriTTS_train_100_RR/{split}.txt'

    with open(read_file, 'r', encoding='utf-8') as f_read:
        for i, line in tqdm(enumerate(f_read)):
            line = line.strip('\n').replace('(None-', '').split(') ')
            id = line[0]
            txt = line[1].split(' ')
            out = [u for i, u in enumerate(txt) if i == 0 or u != txt[i - 1]]  # 去除重复unit
            audio[id] = ' '.join(out)

    f_write = open(write_file, mode='w')
    with open(oldunit_file, 'r', encoding='utf-8') as f_oldunit_file:
        for i, line in tqdm(enumerate(f_oldunit_file)):
            line = line.strip('\n').split('|')
            id = line[0]
            unit = audio[id + '.wav']
            f_write.write(f"{id}|{unit}\n")
