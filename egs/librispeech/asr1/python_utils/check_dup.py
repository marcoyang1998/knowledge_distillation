import json
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input_json", help="waveform json file")

def check(args):

    input_json = args.input_json

    with open(input_json,'r') as f:
        data = json.load(f)['utts']
    for k in data:
        assert len(data[k]['output']) == 1, len(data[k]['output'])

if __name__ == '__main__':
    args = parser.parse_args()
    check(args)