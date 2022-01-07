import os
import json
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--working_dir', help="A working dir containing all the json files that are to be merged")
parser.add_argument('--output_dir', help="where to store the merged big json")

def merge(args):
    working_dir = args.working_dir
    output_dir = args.output_dir
    json_files = glob.glob(working_dir + '/data_kd.*.json')
    #print(json_files)
    new_js = {}
    for js in json_files:
        print(js)
        with open(js,'r') as f:
            data = json.load(f)
        for key in data['utts']:
            new_js[key] = data['utts'][key]
    with open(os.path.join(output_dir, 'data_merged.json'), 'w', encoding='utf-8') as f:
        json.dump({'utts':new_js}, f, indent=4, ensure_ascii=False)
    print("Finish merging. Output stored in {}".format(os.path.join(output_dir, 'data_merged.json')))


if __name__ == '__main__':
    args = parser.parse_args()
    merge(args)
