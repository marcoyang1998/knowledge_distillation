import json
import os
from os.path import join, isdir 
from os import mkdir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--working_dir', type=str)
parser.add_argument('--dataset', type=str)

def prepare(args):
    work_dir = args.working_dir
    dataset = args.dataset
    if "100" in dataset.lower():
        dataset = 'audio_file_json/train-clean-100.json'
    elif "360" in dataset.lower():
        dataset = 'audio_file_json/train-clean-360.json'
    elif "500" in dataset.lower():
        dataset = 'audio_file_json/train-other-500.json'
    else:
        raise NotImplementedError() 
    with open(dataset,'r') as f:
        data = json.load(f)
    count = 0
    for k in data:
        region = k.split('-')[0]
        if not isdir(join(work_dir, region)):
            mkdir(join(work_dir, region))
        spkr = '-'.join(k.split('-')[:-1])

        if os.path.isdir(os.path.join(work_dir, region, spkr)):
            continue
        else:
            os.mkdir(os.path.join(work_dir, region, spkr))
            count += 1
            print("Created: {}".format(os.path.join(work_dir,region, spkr)))
    print("Total {} dirs".format(count))
            

if __name__ == '__main__':
    args = parser.parse_args()
    prepare(args)