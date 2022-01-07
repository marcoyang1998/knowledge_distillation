import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_json', type=str)
parser.add_argument('--original_key', type=str)
parser.add_argument('--new_key', type=str)

def change(args):
    input_json = args.input_json
    original_key = args.original_key
    new_key = args.new_key

    with open(input_json, 'r') as f:
        data = json.load(f)
    
    for k in data['utts']:
        new_file = data['utts'][k]['input'][0]['feat'].replace(original_key, new_key)
        #assert os.path.isfile(new_file), print(new_file)
        data['utts'][k]['input'][0]['feat'] = new_file
    
    with open(input_json,'w',encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print('Finished! Output stored in {}'.format(input_json))

if __name__ == '__main__':
    args = parser.parse_args()
    change(args)