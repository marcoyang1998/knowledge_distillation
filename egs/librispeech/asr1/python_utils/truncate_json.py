import json
import argparse
import os

parser = argparse.ArgumentParser(description='Receive input')
parser.add_argument('--input_json', type=str, help="Input json file, e.g folder/data_w2v2.json")
parser.add_argument('--output_folder', type=str, help="Output folder to store the json file, e.g folder")
parser.add_argument('--truncate_number', type=int, default=0, help="Number of utterances to keep, 0 means keep all")
parser.add_argument('--reverse', type=bool, default=False)

def truncate(input_path, output_path, num, reverse):
    with open(input_path, 'r') as f:
        data = json.load(f)
    truncated_json = {}
    if num == 0:
        num = len(data['utts'])
    i = 0
    keys = list(data['utts'].keys())
    if reverse:
        keys = keys[::-1]
    for k in keys:
        truncated_json[k] = data['utts'][k]
        i += 1
        if i == num:
            break
    final = {'utts':truncated_json}    
    file_name = input_path.split('/')[-1]
    file_name = file_name.replace('.json','_truncated_{}.json'.format(i))
    with open(os.path.join(output_path, file_name), 'w',encoding='utf8') as f:
        json.dump(final, f, indent=4,ensure_ascii=False)
    print("Finish truncating. Output stored in {} with {} utterances".format(os.path.join(output_path, file_name), i))

if __name__=="__main__":
    args = parser.parse_args()
    in_path = args.input_json
    out_path = args.output_folder
    num = args.truncate_number
    reverse = args.reverse
    truncate(in_path, out_path, num, reverse)

