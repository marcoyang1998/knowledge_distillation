import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='python_utils/logs/log', help="input log file")
parser.add_argument('--key', type=str, help="what to plot, error or loss")
parser.add_argument('--average', type=str, default='true', help="moving average of 5")
parser.add_argument('--save', type=str, default='false')

def plot_log(args):
    input_file = args.input
    key = args.key
    average = args.average.lower()
    save = args.save

    with open(input_file, 'r') as f:
        data = json.load(f)

    if key == 'loss':
        key1 = 'main/loss'
        key2 = 'main/loss_kd'
        key3 = 'validation/main/loss'
        keys = [key1, key2, key3]
        pts = {key1: [], key2: [], key3: []}
        iter_plt = []
        for log in data:
            for key in keys:
                it_num = log['iteration']
                item = log[key]
                pts[key].append(item)
            iter_plt.append(it_num)
        for key in keys:
            if average == 'true':
                pts[key].insert(0,pts[key][0])
                pts[key].insert(0,pts[key][0]) # done for padding
                pts[key].append(pts[key][-1])
                pts[key].append(pts[key][-1])
                arr = np.array(pts[key])
                pts[key] = np.convolve(arr, np.ones(5), 'valid') / 5
            plt.plot(iter_plt, pts[key],label=key)
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.grid()
        plt.legend()
        plt.show()


    elif key == 'error':
        key = 'validation/main/cer_ctc'
        iter_plt = []
        dt_plt = []

        for log in data:
            if key in log:
                it_num = log['iteration']
                item = log[key]
                iter_plt.append(it_num)
                dt_plt.append(item)
        if average == 'true':
            dt_plt.insert(0,dt_plt[0])
            dt_plt.insert(0,dt_plt[0])
            dt_plt.append(dt_plt[-1])
            dt_plt.append(dt_plt[-1])
            dt_plt = np.array(dt_plt)
            dt_plt = np.convolve(dt_plt, np.ones(5), 'valid') / 5
        plt.plot(iter_plt, dt_plt, label=key)
        plt.xlabel("iterations")
        plt.ylabel("cer")
        plt.grid()
        plt.legend()
        plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    plot_log(args)

