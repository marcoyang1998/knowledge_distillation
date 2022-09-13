#!/usr/bin/env bash
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
ngpu=8         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
verbose=0      # verbose option

# feature configuration
do_delta=false

train_config=
decode_config=conf/decode.yaml

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=librispeech_clean_100
data_dir=waveform_training_data

model=$1
exptag=$2
expname=train_${train_set}_${model}_${exptag}
expdir=exp_wavlm_bpe/${expname}
echo "exp dir: ${expdir}" 
mkdir -p ${expdir}

train_dir=${data_dir}/train_clean_100_shortened
valid_dir=${data_dir}/dev_other
dict=data/lang_char/train_clean_100_units.txt
echo "dictionary: ${dict}"

resume=

echo "Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${train_dir}/data.json \
        --valid-json ${valid_dir}/data.json