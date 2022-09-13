#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

stage=1
stop_stage=2
nj=8
nbpe=29
#bpemode=char # character level, set to unigram if want to use bpe
dumpdir=dump   # directory to dump full features
do_delta=false

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


datadir=downloads # root folder for librispeech dataset
train_set=train_clean_100
recog_set="test_clean test_other dev_clean dev_other"

dict=data/lang_char/${train_set}_units.txt
echo "dictionary: ${dict}"


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${datadir}/LibriSpeech/${part} data/${part//-/_}
    done
fi

# before proceeding with stage 2, you need to first genenrate a .scp file.
# The each row of scp file should have the following structure
# utterance_ID /path/to/ID.flac
# e.g 
# 1272-128104-0002 /downloads/LibriSpeech/train_clean_100/101/101-001/101-001-0001.flac
# This scp file should be stored under ${dumpdir}/${set}/delta${do_delta}

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}
    #spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --split_by_unicode_script=false
    #spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}

    # make json labels
    feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
    data2json.sh --nj ${nj} --feat ${feat_tr_dir}/feats.scp \
        data/${train_set} ${dict} > ${feat_tr_dir}/data.json

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi