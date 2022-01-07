import json
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--file1", help="f1")
parser.add_argument("--file2", help="f2")

def compare(args):
    file1 = args.file1
    file2 = args.file2
    with open(file1,'r') as f:
        f1 = json.load(f)['utts']
    with open(file2,'r') as f:
        f2 = json.load(f)['utts']
    print("File1: {} utts".format(len(f1)))
    print("File2: {} utts".format(len(f2)))
    #assert len(f1) == len(f2)
    count = 0
    for k in f1:
        if f1[k]['output'][0]['tokenid'] == f2[k]['output'][0]['tokenid']:
            count += 1
    print("{}/{} utts are same".format(count, len(f1)))

if __name__ == '__main__':
    args = parser.parse_args()
    compare(args)