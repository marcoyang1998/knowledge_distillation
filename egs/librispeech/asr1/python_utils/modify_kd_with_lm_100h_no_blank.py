import json
import argparse
import os
from tqdm import tqdm
from os.path import join, isfile, isdir
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_json', type=str)
parser.add_argument('--lm_logits_dir', type=str, help="xx/npy_files")
parser.add_argument('--npy_output_dir', type=str)
parser.add_argument('--json_output_dir', type=str)

def modify(args):
    input_json = args.input_json
    lm_logits_dir = args.lm_logits_dir
    npy_output_dir = args.npy_output_dir
    json_output_dir = args.json_output_dir
    
    with open(input_json,'r') as f:
        data = json.load(f)['utts']
    print('Total {} utts.'.format(len(data)))

    new_js = {}
    for k in tqdm(data):
        npy_file = data[k]['output'][1]['feat']
        kd_data = np.load(npy_file)
        assert kd_data.shape[1] == 261, k
        #with_lm = kd_data[:,2:]
        yseq = kd_data[:,2]
        count_token = 0

        l = kd_data.shape[0]
        region = k.split('-')[0]
        spkr = '-'.join(k.split('-')[:2])
        lm_logit_file = join(lm_logits_dir,region,spkr,k+'.npy')
        assert isfile(lm_logit_file), lm_logit_file
        lm_logit = np.load(lm_logit_file)

        for i in range(l):
            if i ==0:
                continue
            else:
                kd_data[i,3:] = kd_data[i,3:]*1.3 - 0.3*lm_logit[count_token]
            if yseq[i] != 0:
                count_token += 1

        count_token = 0
        for i in range(l):
            #print(with_lm.shape, lm_logit[count_token].shape)
            kd_data[i,4:] = kd_data[i,4:] + 0.3*lm_logit[count_token][1:]
            if yseq[i] != 0:
                count_token += 1

        output_npy = join(npy_output_dir,region,spkr,k+'.npy')
        if not isdir(join(npy_output_dir,region,spkr)):
            os.makedirs(join(npy_output_dir,region,spkr))
        np.save(output_npy, kd_data)
        new_js[k] = data[k]
        new_js[k]['output'][1]['feat'] = output_npy

    out_js = join(json_output_dir, 'data.json')

    with open(out_js,'w',encoding='utf-8') as f:
        json.dump({'utts':new_js}, f, indent=4, ensure_ascii=False)
    print('Finished! Output stored in {}'.format(out_js))

if __name__ == '__main__':
    args = parser.parse_args()
    modify(args)