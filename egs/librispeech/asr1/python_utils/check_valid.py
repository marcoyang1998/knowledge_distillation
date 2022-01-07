import json
import os
import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)

def check_and_fix(args):
    input_file = args.input

    with open(input_file,'r') as f:
        data = json.load(f)['utts']
    new_js = {}
    count = 0
    for k in tqdm(data):
        assert len(data[k]['output']) == 2, k
        #assert os.path.isfile(data[k]['output'][1]['feat']), data[k]['output'][1]['feat']
        #assert data[k]['output'][0]['text'] != "", data[k]
        if data[k]['output'][0]['text'] == "":
            count += 1
            print(k)
            continue
        new_js[k] = data[k]
        #assert data.shape[-1] == 261
        #count += 1
        #kd_array = np.load(data[k]['output'][1]['feat'])
        #if sum(kd_array.reshape(-1) == -1)>0:
        #    print(k)
        #del kd_array
    if count != 0:
        with open(input_file, 'w', encoding='utf8') as f:
            json.dump({'utts': new_js}, f, indent=4,ensure_ascii=False)
        print(f'Output stored in {input_file}.')

    print(f"All checked! A total of {count} training samples are invalid.")
    
if __name__ == '__main__':
    args = parser.parse_args()
    check_and_fix(args)