import json
import argparse
import os
from os.path import isfile
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_json",type=str,help="input json file")

def correct_path(args):
    input_json = args.input_json
    with open(input_json, 'r') as f:
        data = json.load(f)['utts']
    for k in tqdm(data):
        kd_file_name = data[k]["output"][1]['feat']
        region = k.split('-')[0]
        data[k]["output"][1]['feat'] = kd_file_name.replace('npy_files','npy_files/{}'.format(region))
        assert isfile(data[k]["output"][1]['feat']), data[k]["output"][1]['feat']
    
    with open(input_json,'w',encoding='utf-8') as f:
        json.dump({'utts':data}, f, indent=4, ensure_ascii=False)
    print("Finished! Output stored in {}".format(input_json))

if __name__ == '__main__':
    args = parser.parse_args()
    correct_path(args)