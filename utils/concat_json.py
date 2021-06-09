import json
import argparse
import os

parser = argparse.ArgumentParser(description="Concatenate two json files")
parser.add_argument("--file1", type=str, help="First json file")
parser.add_argument("--file2", type=str, help="Second json file")
parser.add_argument("--output_folder", type=str, help="Output folder")

def concat(f1, f2, output_folder):
    with open(f1, 'r') as f:
        data1 = json.load(f)
    with open(f2, 'r') as f:
        data2 = json.load(f)
    print("File1: {} utterances, file2: {} utterances".format(len(data1['utts']), len(data2['utts'])))
    for k in data2["utts"]:
        data1["utts"][k] = data2["utts"][k]
    with open(os.path.join(output_folder, "data_w2v2.json"), 'w') as f:
        json.dump(data1, f)
    print("New json stored in {} with {} utterances".format(os.path.join(output_folder, "data_w2v2.json"), len(data1['utts'])))

if __name__ == '__main__':
    args = parser.parse_args()
    f1 = args.file1
    f2 = args.file2
    output_folder = args.output_folder
    concat(f1,f2, output_folder)