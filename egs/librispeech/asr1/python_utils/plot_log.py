import json
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help="input log file")
parser.add_argument('--key', type=str, help="which key to plot")

def plot_log(args):
    input_file = args.input
    key = args.key

    with open(input_file, 'r') as f:
        data = json.load(f)

    iter_plt = []
    dt_plt = []

    for log in data:
        if key in log:
            it_num = log['iteration']
            item = log[key]
            iter_plt.append(it_num)
            dt_plt.append(item)
        #print(type(l))
    #print(iter_plt, dt_plt)
    plt.plot(iter_plt, dt_plt)
    plt.xlabel("iterations")
    plt.ylabel(key)
    plt.show()









if __name__ == '__main__':
    args = parser.parse_args()
    plot_log(args)
