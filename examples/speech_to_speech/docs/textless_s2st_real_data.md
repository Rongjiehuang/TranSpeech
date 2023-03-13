# Textless Speech-to-Speech Translation (S2ST) on Real Data

We provide instructions and pre-trained models for the work "[Textless Speech-to-Speech Translation on Real Data (Lee et al. 2021)](https://arxiv.org/abs/2112.08352)".

## Pre-trained Models

### HuBERT
Model | Pretraining Data | Model | Quantizer
|---|---|---|---
mHuBERT Base | [VoxPopuli](https://github.com/facebookresearch/voxpopuli) En, Es, Fr speech from the 100k subset | [download](https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt) | [L11 km1000](https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin)


### Unit-based HiFi-GAN vocoder
Unit config | Unit size | Vocoder language | Dataset | Model
|---|---|---|---|---
mHuBERT, layer 11 | 1000 | En | [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) | [ckpt](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000), [config](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json)
mHuBERT, layer 11 | 1000 | Es | [CSS10](https://github.com/Kyubyong/css10) | [ckpt](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10/g_00500000), [config](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10/config.json)
mHuBERT, layer 11 | 1000 | Fr | [CSS10](https://github.com/Kyubyong/css10) | [ckpt](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_fr_css10/g_00500000), [config](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_fr_css10/config.json)


### Speech normalizer
Language | Training data | Target unit config | Model
|---|---|---|---
En | 10 mins | mHuBERT, layer 11, km1000 | [download](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/speech_normalizer/en/en_10min.tar.gz)
En | 1 hr | mHuBERT, layer 11, km1000 | [download](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/speech_normalizer/en/en_1h.tar.gz)
En | 10 hrs | mHuBERT, layer 11, km1000 | [download](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/speech_normalizer/en/en_10h.tar.gz)
Es | 10 mins | mHuBERT, layer 11, km1000 | [download](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/speech_normalizer/es/es_10min.tar.gz)
Es | 1 hr | mHuBERT, layer 11, km1000 | [download](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/speech_normalizer/es/es_1h.tar.gz)
Es | 10 hrs | mHuBERT, layer 11, km1000 | [download](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/speech_normalizer/es/es_10h.tar.gz)
Fr | 10 mins | mHuBERT, layer 11, km1000 | [download](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/speech_normalizer/fr/fr_10min.tar.gz)
Fr | 1 hr | mHuBERT, layer 11, km1000 | [download](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/speech_normalizer/fr/fr_1h.tar.gz)
Fr | 10 hrs | mHuBERT, layer 11, km1000 | [download](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/speech_normalizer/fr/fr_10h.tar.gz)

* Refer to the paper for the details of the training data.

## Train a Speech Normalizer
1. Get Pseudo label using the learned clusters
```bash
MANIFEST=<tab_separated_manifest_of_audio_files_to_quantize>
OUT_QUANTIZED_FILE=<output_quantized_audio_file_path>

python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension ".flac"
```

2. Fine-tune a HuBERT model with a CTC loss
```bash
Suppose {train,valid}.tsv are saved at /path/to/data, and their corresponding character transcripts {train,valid}.ltr are saved at /path/to/trans.

To fine-tune a pre-trained HuBERT model at /path/to/checkpoint, run

$ python fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq-py/examples/hubert/config/finetune \
  --config-name base_10h_change \
  task.data=/path/to/data task.label_dir=/path/to/trans \
  model.w2v_path=/path/to/checkpoint
```



## Inference with Pre-trained Models

### Speech normalizer
1. Download the pre-trained models, including the dictionary, to `DATA_DIR`.
2. Format the audio data.
```bash
# AUDIO_EXT: audio extension, e.g. wav, flac, etc.
# Assume all audio files are at ${AUDIO_DIR}/*.${AUDIO_EXT}

python examples/speech_to_speech/preprocessing/prep_sn_data.py \
  --audio-dir ${AUDIO_DIR} --ext ${AUIDO_EXT} \
  --data-name ${GEN_SUBSET} --output-dir ${DATA_DIR} \
  --for-inference
```

3. Run the speech normalizer and post-process the output.
```bash
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

CUDA_VISIBLE_DEVICES=0 python examples/speech_recognition/new/infer.py  
--config-dir examples/hubert/config/decode/ --config-name infer_viterbi 
task.data=/home/huangrongjie/Project/Translation/fairseq/data/processed_fr_textless/tsvdir/  
task.normalize=false  
common_eval.results_path=/home/huangrongjie/Project/Translation/fairseq/speech2unit/HuBERTs/TGT_AUDIO_fr_RR/testset_inferresults/ 
common_eval.path=/home/huangrongjie/Project/Translation/fairseq/speech2unit/HuBERTs/TGT_AUDIO_fr_RR/checkpoints/checkpoint_best.pt 
dataset.gen_subset=test '+task.labels=["unit"]' 
+decoding.results_path=/home/huangrongjie/Project/Translation/fairseq/speech2unit/HuBERTs/TGT_AUDIO_fr_RR/units/ 
common_eval.post_process=none +dataset.batch_size=1 common_eval.quiet=True

# Post-process and generate output at ${RESULTS_PATH}/${GEN_SUBSET}.txt
python examples/speech_to_speech/preprocessing/prep_sn_output_data.py \
  --in-unit ${RESULTS_PATH}/hypo.units \
  --in-audio ${DATA_DIR}/${GEN_SUBSET}.tsv \
  --output-root ${RESULTS_PATH}

# generate output at ${RESULTS_PATH}/${GEN_SUBSET}.tsv
Set --reduce-unit for training S2UT reduced model
Pre-trained vocoder and config ($VOCODER_CKPT, $VOCODER_CFG) can be downloaded from the Pretrained Models section. They are not required if --eval-inference is not going to be set during model training.
# $SPLIT1, $SPLIT2, etc. are split names such as train, dev, test, etc.

python examples/speech_to_speech/preprocessing/prep_s2ut_data.py \
  --source-dir $SRC_AUDIO --target-dir $TGT_AUDIO --data-split $SPLIT1 $SPLIT2 \
  --output-root $DATA_ROOT --reduce-unit \
  --vocoder-checkpoint $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG


python examples/speech_to_speech/preprocessing/prep_s2ut_data.py 
--source-dir data/SRC_AUDIO_fr/ --target-dir speech2unit/HuBERTs/TGT_AUDIO_fr_RR/units/
--data-split train test dev 
--output-root  speech2unit/HuBERTs/TGT_AUDIO_fr_RR/processed_units/ --reduce-unit 
--vocoder-checkpoint  vocoder/g_00500000 --vocoder-cfg vocoder/config.json

```





### Unit-to-waveform conversion with unit vocoder
The pre-trained vocoders can support generating audio for both full unit sequences and reduced unit sequences (i.e. duplicating consecutive units removed). Set `--dur-prediction` for generating audio with reduced unit sequences.
```bash
# IN_CODE_FILE contains one unit sequence per line. Units are separated by space.

python examples/speech_to_speech/generate_waveform_from_code.py \
  --in-code-file ${IN_CODE_FILE} \
  --vocoder ${VOCODER_CKPT} --vocoder-cfg ${VOCODER_CFG} \
  --results-path ${RESULTS_PATH} --dur-prediction
```

## Training new models
To be updated.
