import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_json',type=str)
parser.add_argument('--output_folder',type=str)

def keep_trans(args):
    input_json = args.input_json
    output_folder = args.output_folder

    with open(input_json,'r') as f:
        data = json.load(f)['utts']

    new_js = {}
    for key in data:
        new_js[key] = data[key]
        new_js[key]['output'] = [data[key]['output'][0]]
    output_file = os.path.join(output_folder, 'data_trans.json')

    with open(output_file, 'w',encoding='utf8') as f:
            json.dump({'utts':new_js}, f, indent=4, ensure_ascii=False)
    print("Output stored in {}".format(output_file))


if __name__ == '__main__':
    args = parser.parse_args()
    keep_trans(args)
        