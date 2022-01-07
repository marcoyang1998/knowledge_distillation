import json
import os
import glob
import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str)
parser.add_argument('--set', type=str, help='train-clean-100 or ...')
parser.add_argument('--ext', type=str, default='flac')

def generate_filelist(args):
    root_dir = args.root_dir
    set = args.set
    ext = args.ext
    if set == 'train-clean-100':
        audio_files = glob.glob(root_dir + '/' + set + '/' + '/*/*/*.' + ext, recursive=True)
    else:
        audio_files = glob.glob(root_dir + '/' + set + '/' + '/*/*/*.' + ext, recursive=True)
    
    data = {}
    count = 0
    for audio in audio_files:
        assert os.path.isfile(audio), "Not found: {}".format(audio)
        key = audio.split('/')[-1].split('.')[0]
        #print(key)
        data[key] = audio
        count += 1

    print("Total count: {} audio files".format(count))

    output_name = os.path.join('audio_file_json', set+'.json')
    with open(output_name, 'w') as f:
        json.dump(data, f, indent=4)
    print('Data stored in {}'.format(output_name))


if __name__ == '__main__':
    args = parser.parse_args()
    generate_filelist(args)