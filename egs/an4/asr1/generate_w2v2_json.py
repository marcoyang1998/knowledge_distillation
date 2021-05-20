import json
import argparse
import os
parser = argparse.ArgumentParser(description='Process the original train/dev/test json to be used ')
parser.add_argument('filetype', type=str, default='wav', help='file type of the audio')
parser.add_argument('json_path',type=str)
parser.add_argument('dataset_dir',type=str)
parser.add_argument('dataset', type=str, help="train/dev/test")

def convert(args):
    with open(args.json_path, 'r') as f:
        train = json.load(f)
    for key in train["utts"]:
        folder_name, name, suffix = key.split('-')
        train["utts"][key]["input"][0]["feat"] = os.path.join(args.dataset_dir, folder_name, name+'-'+folder_name+'-'+suffix+'.'+ args.filetype)
        train["utts"][key]["input"][0]["filetype"] = args.filetype

    with open("{}.json".format(args.dataset), 'w') as f:
        json.dump(train, f)


if __name__=="__main__":
    args = parser.parse_args()
    convert(args)
    
