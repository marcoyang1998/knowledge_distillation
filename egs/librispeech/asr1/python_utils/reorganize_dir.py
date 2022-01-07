import os
from os.path import join, isfile, isdir
from os import listdir, mkdir
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--working_dir",type=str,help="The dir you want to re-organize")
parser.add_argument("--organize_type", type=str, help="region or number")

def reorganize(args):
    working_dir = args.working_dir
    dir_list = [f for f in listdir(working_dir) if isdir(join(working_dir,f))]
    print("Original folder contains {} sub-directories".format(len(dir_list)))
    for f in dir_list:
        region = f.split('-')[0]
        if isdir(join(working_dir,region)):
            pass
        else:
            mkdir(join(working_dir, region))
        shutil.move(join(working_dir,f),join(working_dir,region))
    new_dir_list = [f for f in listdir(working_dir) if isdir(join(working_dir,f))]
    print("{} folders after reorganizing.".format(len(new_dir_list)))

def further_reorganize(args):
    working_dir = args.working_dir
    dir_list = [f for f in listdir(working_dir) if isdir(join(working_dir,f))]
    print("Original folder contains {} sub-directories".format(len(dir_list)))
    for f in dir_list:
        new_folder = join(working_dir,'f_{}'.format(f[0]))
        if isdir(new_folder):
            pass
        else:
            mkdir(new_folder)
        shutil.move(join(working_dir,f),new_folder)
    new_dir_list = [f for f in listdir(working_dir) if isdir(join(working_dir,f))]
    print("{} folders after reorganizing.".format(len(new_dir_list)))

if __name__ == '__main__':
    args = parser.parse_args()
    if "region" in args.organize_type.lower():
        reorganize(args)
    elif "number" in args.organize_type.lower():
        further_reorganize(args)
    else:
        raise NotImplementedError()
