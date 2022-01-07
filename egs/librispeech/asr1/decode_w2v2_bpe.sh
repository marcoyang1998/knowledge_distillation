#!/usr/bin/env bash
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=5       # start from -1 if you need to start from data download
stop_stage=100
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=1
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

#dumpdir=dump_bpe
decode_config=conf/decode_ctc_1.0_ins0.25_beam8_lm0.yaml

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

data_len=train_clean_100


expdir=$1
ckpt=$2
nbpe=256

valid_dir=waveform_training_data/
if [ "${ckpt}" == "best" ]; then
    recog_model=model.loss.best
    echo "Recog model: ${recog_model}"
else
    recog_model=snapshot.iter.${ckpt}
    echo "Recog model: ${recog_model}"
fi

dict=data/lang_char/${data_len}_unigram256_units.txt
bpe_model=data/lang_char/${data_len}_unigram256.model
#rnnlm=/home/seekingeta/marcoyang/espnet/egs/librispeech/asr1/exp/train_lm_pytorch_GPT2LM_init2e-4_minscale1e-4_bs128_continue_unigram256_ngpu1/snapshot.iter.980000
#kd_json_dir=rnnt_kd_training_data/bpe256_100h/fbank/train_clean_360

#recog_set="test_clean test_other dev_clean dev_other"
#recog_set="dev_other dev_clean"
recog_set="dev_other_500"
#recog_set="dev_clean"
#recog_set="test_other"
#recog_set=train_clean_100
#recog_set=train_other_500
#recog_set="dev_other_1000"
### you must set batchsize to 0/1 to receive proper decoding results for w2v2!!! ###
lang_model=/home/seekingeta/marcoyang/espnet/egs/librispeech/asr1/exp/train_lm_pytorch_GPT2LM_init2e-4_minscale1e-4_bs128_continue_unigram256_ngpu1/snapshot.iter.1070000

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    pids=()
    for rtask in ${recog_set}; do
        (
            decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${ckpt}
            echo ${decode_dir}
            mkdir -p ${expdir}/${decode_dir}/log/
            feat_recog_dir=${valid_dir}/${rtask}/
            # split data
            splitjson.py --parts ${nj} ${feat_recog_dir}/data.json
            
            ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
                asr_recog.py \
                --config ${decode_config} \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --batchsize 0 \
                --debugmode ${debugmode} \
                --verbose ${verbose} \
                --rnnlm ${lang_model} \
                --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
                --result-label ${expdir}/${decode_dir}/data.JOB.json \
                --model ${expdir}/results/${recog_model} \

            score_sclite.sh --wer true --bpe ${nbpe} --bpemodel ${bpe_model} ${expdir}/${decode_dir} ${dict}
        ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi
echo "Finished"