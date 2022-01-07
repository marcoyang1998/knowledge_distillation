from hashlib import new
import os
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Receive input')
parser.add_argument('--input_json',type=str)

def correct(args):
    input_json = args.input_json
    with open(input_json,'r') as f:
        data = json.load(f)['utts']
    for k in tqdm(data):
        kd_file = data[k]['output'][1]['feat']
        region = k.split('-')[0]
        old_key = 'f_'+region[0]
        #new_key = "npy_files/f_{}/{}".format(region[0], region)
        kd_file = kd_file.replace(old_key, "")
        #print(kd_file)
        assert os.path.isfile(kd_file), kd_file
        data[k]['output'][1]['feat'] = kd_file
        #break
    
    with open(input_json,'w',encoding='utf-8') as f:
        json.dump({'utts': data}, f, indent=4, ensure_ascii=False)
    print('Finished! Output stored in {}'.format(input_json))

if __name__ == '__main__':
    args = parser.parse_args()
    correct(args)
