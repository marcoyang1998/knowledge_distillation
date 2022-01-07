from espnet.nets.pytorch_backend.transformer.embedding import LegacyRelPositionalEncoding
import numpy as np
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input', type=str)

def count(args):
    input_file = args.input
    
    with open(input_file,'r') as f:
        data = json.load(f)['utts']
    
    length_dict = {}
    for key in data:
        length = data[key]["input"][0]["shape"][0]
        if length in length_dict:
            length_dict[length] += 1
        else:
            length_dict[length] = 1
    
    all_length = list(length_dict.keys())
    all_length = sorted(all_length, reverse=True)
    for i,l in enumerate(all_length):
        print(f'{l}: {length_dict[l]}')
        i += 1
        if i > 10:
            break
    
def remove_long_token(args):
    input_file = args.input
    max_len = 275840
    with open(input_file,'r') as f:
        data = json.load(f)['utts']
        
    new_js = {}
    count = 0
    for key in data:
        if data[key]["input"][0]["shape"][0] > max_len:
            count += 1
            continue
        new_js[key] = data[key]
    new_js = {'utts': new_js}
    
    print(f'A total of {count} utts have been removed')
    out_name = input_file.replace('.json',f'shorted{max_len}.json')
    with open(out_name, 'w') as f:
        json.dump(new_js, f, indent=4, ensure_ascii=False)
    print(f'Saved in {out_name}')
        
if __name__=='__main__':
    args = parser.parse_args()
    remove_long_token(args)