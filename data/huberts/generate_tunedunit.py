import pandas as pd
from tqdm import tqdm
from examples.speech_to_text.data_utils import save_df_to_tsv


MANIFEST_COLUMNS = ["id", "src_audio", "src_n_frames", "tgt_audio", "tgt_n_frames"]
audio = {}
for split in ['train', 'test', 'dev']:
    read_file = f'speech2unit/HuBERTs/Tuned_Huberts_011/LibriVox_Fr_En_{split}set_inferresults/hypo.units'
    oldunit_file = f'data/Librivox_S2S/Fr-En/fr_manifest_1000_1/{split}.tsv'
    write_file = f'data/Librivox_S2S/Fr-En/processed_fr_huberts_12/{split}.tsv'
    manifest = {c: [] for c in MANIFEST_COLUMNS}

    with open(read_file, 'r', encoding='utf-8') as f_read:
        for i, line in tqdm(enumerate(f_read)):
            line = line.strip('\n').split(') ')
            id = line[0].split('-')[1]
            txt = line[1].split(' ')
            out = [u for i, u in enumerate(txt) if i == 0 or u != txt[i - 1]]  # 去除重复unit
            audio[id] = ' '.join(out)


    with open(oldunit_file, 'r', encoding='utf-8') as f_manifest:
        for i, line in tqdm(enumerate(f_manifest)):
            if i == 0: continue
            line = line.strip('\n').split('\t')
            id = line[0] + '.wav'
            manifest["id"].append(line[0])
            manifest["src_audio"].append(line[1])
            manifest["src_n_frames"].append(line[2])
            manifest["tgt_audio"].append(audio[id])
            manifest["tgt_n_frames"].append(line[4])

    print(f"Writing manifest to {write_file}...")
    save_df_to_tsv(pd.DataFrame.from_dict(manifest), write_file)
