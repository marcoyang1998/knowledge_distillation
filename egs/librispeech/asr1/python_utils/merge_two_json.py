import json
import argparse
import glob
from tqdm import tqdm
from os.path import join
from os.path import isfile
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--full_json', type=str, help="original kd fbank json file")
parser.add_argument('--incomplete_json', type=str, help="incomplete kd json file")
parser.add_argument('--output_dir', type=str)

def perform(args):
    full_json = args.full_json
    incomplete_json = args.incomplete_json
    output_dir = args.output_dir

    with open(full_json,'r') as f:
        full_json = json.load(f)['utts']
    print(len(full_json))

    with open(incomplete_json,'r') as f:
        incomplete_json = json.load(f)['utts']

    print(len(incomplete_json))

    new_js = {}

    for k in full_json:
        if k in incomplete_json:
            new_js[k] = incomplete_json[k]
        else:
            new_js[k] = full_json[k]

    output_name = join(output_dir,'data_kd_mixed.json')
    with open(output_name, 'w',encoding='utf8') as f:
            json.dump({'utts':new_js}, f, indent=4, ensure_ascii=False)
    print("Output stored in {} with {} utts".format(output_name, len(new_js)))

if __name__ == '__main__':
    args = parser.parse_args()
    perform(args)