#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=11
stop_stage=13

train_set="train"
valid_set="valid"
test_sets="devman devsge"

lm_config=conf/tuning/train_lm_transformer.yaml
asr_config=conf/whisper/train_asr_whisper_small_adapter_encoder.yaml
inference_config=conf/whisper/decode_asr_whisper.yaml

if [ ! -f "data/train/token.man.2" ]; then
    # must preprocess data first to get Mandarin character tokens
    if [ ${stage} -eq 1 ]; then
        ./asr.sh --stage 1 --stop_stage 1
        stage=2
    else
        echo "Error: data/train/token.man.2 does not exist! Run from stage=1 again."
        exit 1
    fi
fi
man_chars=2622
bpe_nlsyms=""
source data/train/token.man.2  # for bpe_nlsyms & man_chars
nbpe=$((3000 + man_chars + 4))  # 5626
# English BPE: 3000 / Mandarin: 2622 / other symbols: 4

freeze_param="adapter"
use_lm=false
use_wordlm=false
# pre_trained_weight=/home/espnet/egs2/seame/asr1/exp/asr_seame_whisper_small_zh_en_adapter_encoder_nocsloss/valid.acc.ave.pth
./asr.sh \
    --ngpu 1 \
    --nj 5 \
    --gpu_inference true \
    --inference_nj 1 \
    --ngpu 1 \
    --audio_format "flac.ark" \
    --feats_type raw \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --token_type whisper_multilingual \
    --feats_normalize '' \
    --cleaner whisper_basic \
    --use_lm ${use_lm}                  \
    --use_word_lm ${use_wordlm}         \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --min_wav_duration 1 \
    --max_wav_duration 30 \
    --asr_speech_fold_length 800 \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text.eng.bpe" \
    --asr_tag "seame_whisper_small_zh_en_adapter_encoder_nocsloss" \
    --score_opts "-e utf-8 -c NOASCII" \
    --freeze_param "${freeze_param}" \
    --pretrained_model ${pre_trained_weight} \
    --ignore_init_mismatch "true" \
    "$@"
    # --pretrained_model ${pre_trained_config} \
