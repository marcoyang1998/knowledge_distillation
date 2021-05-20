#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

asr_config=conf/train_w2v2_new.yaml
lm_config=conf/train_lm_rnn.yaml
inference_config=conf/decode_w2v2.yaml

./asr.sh \
    --lang en \
    --train_set train_nodev \
    --lm_config conf/train_lm.yaml \
    --valid_set train_dev \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --test_sets "train_dev test" \
    --lm_train_text "data/train_nodev/text" "$@"
