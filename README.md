# TranSpeech: Speech-to-Speech Translation With Bilateral Perturbation
#### Rongjie Huang*, Jinglin Liu*, Huadai Liu*, Yi Ren, Lichao Zhang, Jinzheng He, Zhou Zhao | Zhejiang University, ByteDance


PyTorch Implementation of [TranSpeech (ICLR'23)](https://arxiv.org/abs/2205.12523): a speech-to-speech translation model towards high-accuracy and non-autoregressive translation.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2205.12523)
[![GitHub Stars](https://img.shields.io/github/stars/Rongjiehuang/TranSpeech?style=social)](https://github.com/Rongjiehuang/TranSpeech)
![visitors](https://visitor-badge.glitch.me/badge?page_id=Rongjiehuang/TranSpeech)

We provide our implementation and pretrained models in this repository.

Visit our [demo page](https://transpeech.github.io/) for audio samples.

## News
#### TranSpeech is one of our continuous efforts to reduce communication barrier.
- July, 2022: **[TranSpeech](https://arxiv.org/abs/2205.07211)** released at Arxiv.
- March, 2023: **[TranSpeech](https://arxiv.org/abs/2205.07211) (ICLR 2023)** released at Github.
- March, 2023: Audio-Visual Speech-To-Text Translation **[MixSpeech](https://arxiv.org/abs/2303.05309)** and [dataset]() released at Arxiv and Github.

### Dependencies

* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq version 1.0.0a0** and develop locally:
``` bash
pip install --editable ./
```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

# Train your own model

## Data preparation

1. Prepare two folders, `$SRC_AUDIO` and `$TGT_AUDIO`, with `${SPLIT}/${SAMPLE_ID}.wav` for source and target speech under each folder, separately. Note that for S2UT experiments, target audio sampling rate should be in 16,000 Hz, and for S2SPECT experiments, target audio sampling rate is recommended to be in 22,050 Hz.
2. To prepare target discrete units for S2UT model training, see [Generative Spoken Language Modeling (speech2unit)](https://github.com/pytorch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit) for pre-trained k-means models, checkpoints, and instructions on how to decode units from speech. Set the output target unit files (`--out_quantized_file_path`) as `${TGT_AUDIO}/${SPLIT}.txt`. In [Lee et al. 2021](https://arxiv.org/abs/2107.05604), we use 100 units from the sixth layer (`--layer 6`) of the HuBERT Base model.

## Hubert CTC Finetuning 

### 1. Prepare a pretrained Hubert and HifiGAN

Model | Pretraining Data                                                                                 | Model | Quantizer
|---|--------------------------------------------------------------------------------------------------|---|---
mHuBERT Base | [VoxPopuli](https://github.com/facebookresearch/voxpopuli) En, Es, Fr speech from the 100k subset | [download](https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt) | [L11 km1000](https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin)
HIFIGAN | 16k Universal                                                                                    | [download](https://zjueducn-my.sharepoint.com/:f:/g/personal/rongjiehuang_zju_edu_cn/EvMZ_WMcSoVDtUvE-C3wGhoBz4yI_N1Hcfk-LhzVnYMsvg?e=z59ntY)
dict.unit.txt (mHuBERT Fine-tuning)|                                                                                                  | [download](https://zjueducn-my.sharepoint.com/:t:/g/personal/rongjiehuang_zju_edu_cn/Ea5b_NwrBdNGlmNOun6V84sBGdAvFrl1ob2QrBwTYSDSYw?e=Rua4mN)

### 2. Bilateral Perturbation
Suppose we have original dataset at ```/path/to/TGT_AUDIO```

- style normalization: refer to ```./hubertCTC/gen_SN.py``` and generate Dataset S1:
```
python research/TranSpeech/hubertCTC/gen_SN.py  --wav /path/to/TGT_AUDIO --out /path/to/S1/dataset
```
- information enhancement: refer to ```./hubertCTC/gen_IE.py``` and generate Dataset S2
```
python research/TranSpeech/hubertCTC/gen_IE.py --ckpt /path/to/ckpt --wav /path/to/TGT_AUDIO --out /path/to/S2/dataset
```

### 3. Prepare Pseudo Text

- Get Manifest
```
python examples/wav2vec/wav2vec_manifest.py /path/to/S2/dataset --dest /manifest/to/S2/dataset --ext $ext --valid-percent $valid
```
$ext should be set to flac, wav, or whatever format your dataset happens to use that soundfile can read.
$valid should be set to some reasonable percentage (like 0.01) of training data to use for validation.

- Quantize using the learned clusters
```
MANIFEST=/manifest/to/S2/dataset
OUT_QUANTIZED_FILE=/quantized/to/S2/dataset
For CKPT_PATH & KM_MODEL_PATH, refer to Section 1.

python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type hubert \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $CKPT_PATH \
    --layer 11 \
    --manifest_path $MANIFEST  \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension ".flac"
```

### 4. Fine-tune a HuBERT model with a CTC loss
- Prepare {train,valid}.unit

- Get Manifest
```
python data/huberts/generate_tunehuberts.py --manifest /manifest/to/S2/dataset --txt /quantized/to/S2/dataset --unit /unit/to/S2/dataset
```
Suppose we have a mHuBERT Base ckpt at ```/path/to/checkpoint```
Suppose {train,valid}.tsv are saved at ```/manifest/to/S2/dataset```, and their corresponding character transcripts {train,valid}.unit and [dict.unit.txt](https://zjueducn-my.sharepoint.com/:t:/g/personal/rongjiehuang_zju_edu_cn/Ea5b_NwrBdNGlmNOun6V84sBGdAvFrl1ob2QrBwTYSDSYw?e=Rua4mN) are saved at ```/unit/to/S2/dataset```.

- To fine-tune a pre-trained HuBERT model at /path/to/checkpoint, run
```
$ python fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq-py/examples/hubert/config/finetune \
  --config-name base_10h_change \
  task.data=/manifest/to/S2/dataset task.label_dir=/unit/to/S2/dataset \
  model.w2v_path=/path/to/checkpoint optimization.max_update=70000
```

### 5. Inference with Tuned Huberts

- Format the audio data.

```
AUDIO_EXT: audio extension, e.g. wav, flac, etc.
Assume all audio files are at ${AUDIO_DIR}/*.${AUDIO_EXT}
${GEN_SUBSET} should be train, test, or dev

python examples/speech_to_speech/preprocessing/prep_sn_data.py \
  --audio-dir /path/to/TGT_AUDIO --ext ${AUIDO_EXT} \
  --data-name ${GEN_SUBSET} --output-dir ${DATA_DIR} \
  --for-inference
 ```

- Run the Tuned Huberts.
 ```
 mkdir -p ${RESULTS_PATH}
 
python examples/speech_recognition/new/infer.py \
    --config-dir examples/hubert/config/decode/ \
    --config-name infer_viterbi \
    task.data=${DATA_DIR} \
    task.normalize=false \
    common_eval.results_path=${RESULTS_PATH}/log \
    common_eval.path=${DATA_DIR}/checkpoint_best.pt \
    dataset.gen_subset=${GEN_SUBSET} \
    '+task.labels=["unit"]' \
    +decoding.results_path=${RESULTS_PATH} \
    common_eval.post_process=none \
    +dataset.batch_size=1 \
    common_eval.quiet=True
 ```

- Post-process and generate output at ${RESULTS_PATH}/${GEN_SUBSET}.txt
 ```
python examples/speech_to_speech/preprocessing/prep_sn_output_data.py \
  --in-unit ${RESULTS_PATH}/hypo.units \
  --in-audio ${DATA_DIR}/${GEN_SUBSET}.tsv \
  --output-root ${RESULTS_PATH}
 ```


### 6. Formatting Speech-to-Speech Translation data

```
# $SPLIT1, $SPLIT2, etc. are split names such as train, dev, test, etc.

python examples/speech_to_speech/preprocessing/prep_s2ut_data.py \
  --source-dir $SRC_AUDIO --target-dir $TGT_AUDIO --data-split $SPLIT1 $SPLIT2 \
  --output-root $DATA_ROOT --reduce-unit \
  --vocoder-checkpoint $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG
```

For knowledge distillation, we need another step to format the data from teacher.


## Training S2UT model

Here's an example for training nar_s2ut_conformer S2UT models with 1000 discrete units as target:
```
fairseq-train $DATA_ROOT \
  --config-yaml config.yaml \
  --task speech_to_speech_fasttranslate --target-is-code --target-code-size 1000 --vocoder code_hifigan  \
  --criterion nar_speech_to_unit --label-smoothing 0.2 \
  --arch nar_s2ut_conformer --share-decoder-input-output-embed \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset train --valid-subset dev \
  --save-dir ${MODEL_DIR}  --tensorboard-logdir ${MODEL_DIR} \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 10000 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 10.0 \
  --max-update 400000 --max-tokens 20000 --max-target-positions 3000 --update-freq 4 \
  --seed 1 --fp16 --num-workers 8 \
  --user-dir research/  --attn-type espnet --pos-enc-type rel_pos 
```
* Adjust `--update-freq` accordingly for different #GPUs. In the above we set `--update-freq 4` to simulate training with 4 GPUs.

## Inference with NAR S2UT model

1. Follow the same inference process as in fairseq-S2T to generate unit sequences (${RESULTS_PATH}/generate-${GEN_SUBSET}.txt).
```
fairseq-generate $DATA_ROOT \
 --gen-subset test --task speech_to_speech_fasttranslate  --path ${MODEL_DIR} \
 --target-is-code --target-code-size 1000 --vocoder code_hifigan   --results-path ${OUTPUT_DIR} \
 --iter-decode-max-iter $N  --iter-decode-eos-penalty 0 --beam 1   --iter-decode-with-beam 15 
```

2. Convert unit sequences to waveform.
```
grep "^D\-" ${RESULTS_PATH}/generate-${GEN_SUBSET}.txt | \
  sed 's/^D-//ig' | sort -nk1 | cut -f3 \
  > ${RESULTS_PATH}/generate-${GEN_SUBSET}.unit

grep "^T\-" ${RESULTS_PATH}/generate-${GEN_SUBSET}.txt  | \
  sed 's/^T-//ig' | sort -nk1 | cut -f2 
  > ${RESULTS_PATH}/ref-${GEN_SUBSET}.unit
```
 * Set `--dur-prediction` for generating audio for S2UT _reduced_ models.
 


* Noisy decoding: inference with `--external-reranker  --path ${checkpoint_path} = a:b `, where `a, b` denote the student and AR tracher.

## Unit-to-Speech HiFi-GAN vocoder

Unit config | Unit size | Vocoder language | Dataset | Model
|---|---|---|---|---
mHuBERT, layer 11 | 1000 | En | [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) | [ckpt](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000), [config](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json)
mHuBERT, layer 11 | 1000 | Es | [CSS10](https://github.com/Kyubyong/css10) | [ckpt](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10/g_00500000), [config](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10/config.json)
mHuBERT, layer 11 | 1000 | Fr | [CSS10](https://github.com/Kyubyong/css10) | [ckpt](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_fr_css10/g_00500000), [config](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_fr_css10/config.json)

```
python examples/speech_to_speech/generate_waveform_from_code.py \
  --in-code-file ${RESULTS_PATH}/generate-${GEN_SUBSET}.unit \
  --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
  --results-path ${RESULTS_PATH} --dur-prediction
```

## Evaluation
Refer to [research/TranSpeech/asr_bleu/README.md](research/TranSpeech/asr_bleu/README.md)


## Acknowledgements
This implementation uses parts of the code from the following Github repos:
[Fairseq](https://github.com/facebookresearch/fairseq),
as described in our code.

## Citations ##
If you find this code useful in your research, please cite our work:
```bib
@article{huang2022transpeech,
  title={TranSpeech: Speech-to-Speech Translation With Bilateral Perturbation},
  author={Huang, Rongjie and Zhao, Zhou and Liu, Jinglin and Liu, Huadai and Ren, Yi and Zhang, Lichao and He, Jinzheng},
  journal={arXiv preprint arXiv:2205.12523},
  year={2022}
}
```

## Disclaimer ##
Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.

