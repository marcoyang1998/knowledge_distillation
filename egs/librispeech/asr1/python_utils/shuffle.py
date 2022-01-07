import json
import numpy as np
import argparse
import os

parser= argparse.ArgumentParser()
parser.add_argument('--input_json', type=str, help="Input json file")

def shuffle_json(args):
    input_json = args.input_json
    with open(input_json,'r') as f:
        data = json.load(f)
    num_utts = len(data['utts'])
    keys = list(data['utts'].keys())
    indices = np.arange(num_utts)
    np.random.shuffle(indices)

    new_js = {}
    count = 0
    for ind in indices:
        new_js[keys[ind]] = data['utts'][keys[ind]]
        count += 1

    assert count == num_utts

    folder = '/'.join(input_json.split('/')[:-1])
    file_name = input_json.split('/')[-1].split('.')[0]
    with open(os.path.join(folder, file_name+'_shuffled.json'), 'w') as f:
        json.dump({'utts': new_js}, f, indent=4)
    print('Output stored in {}'.format(os.path.join(folder, file_name+'_shuffled.json')))

if __name__=="__main__":
    args = parser.parse_args()
    shuffle_json(args)