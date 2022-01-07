#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=4       # start from -1 if you need to start from data download
stop_stage=100
ngpu=8         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/train_cfms_100h_kd.yaml

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

model=$1
exptag=$2
# datalen: clean_100 or mix_460 or ...
datalen=$3
# label_type: gt or reduced_lattice
label_type=$4

train_set=librispeech_${datalen}h_${label_type}
expname=train_${train_set}_${backend}_${model}_${exptag}
expdir=exp_conformer_rnnt_kd_finetune/${expname}
echo "exp name: ${expdir}" 
mkdir -p ${expdir}

train_dir=rnnt_kd_training_data/bpe256_100h/fbank/train_${datalen}_${label_type}_label
valid_dir=${dumpdir}/dev_other_100/delta${do_delta}
dict=data/lang_char/train_clean_100_unigram256_units.txt
echo "dictionary: ${dict}"

resume=/home/seekingeta/marcoyang/espnet/egs/librispeech/asr1/exp_conformer/train_librispeech_100h_pytorch_cfms_transducer_bpe256_specaug_subsample4/results/snapshot.iter.121600
echo "resume from : ${resume}"

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
	    --preprocess-conf ${preprocess_config} \
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
	    --train-json ${train_dir}/data_kd.json \
        --valid-json ${valid_dir}/data_unigram256_truncated_100.json
fi
