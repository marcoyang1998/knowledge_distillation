import soundfile
import os
import numpy as np
import glob
from tqdm import tqdm
import random

def time_check():
    working_dir = "/rds/user/xy316/hpc-work/mphil/espnet/egs/librispeech/asr1/rnnt_label/train_clean_100_bpe256/npy_files"
    #working_dir = "/rds/user/xy316/hpc-work/mphil/espnet/egs/librispeech/asr1/rnnt_label/train_clean_360_bpe256_copy/npy_files"
    audio_files = glob.glob(working_dir + '/*/*/*.npy', recursive=True)
    random.Random(4).shuffle(audio_files)
    print(audio_files[:10])
    for f in tqdm(audio_files):
        audio = np.load(f)
        del audio

if __name__ == '__main__':
    time_check()