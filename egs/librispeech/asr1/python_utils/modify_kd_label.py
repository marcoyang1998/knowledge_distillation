import json
import numpy as np
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_json',type=str)

def add_tu_list_to_kd_label(args):
    input_json = args.input_json
    with open(input_json, 'r') as f:
        data = json.load(f)['utts']

    for k in tqdm(data):
        kd_file = data[k]["output"][1]['feat']
        assert os.path.isfile(kd_file), k
        kd_array = np.load(kd_file)
        if kd_array.shape[-1] == 261:
            data[k]["output"][1]['shape'] = [kd_array.shape[0], kd_array.shape[1]]
            continue
        #assert kd_array.shape[-1] == 259, kd_array.shape
        seq_with_blank = kd_array[:,0]
        t_list = []
        u_list = []
        t,u =0,0
        for token in seq_with_blank:
            t_list.append(t)
            u_list.append(u)
            if token > 0:
                u += 1
            else:
                t += 1
        #assert len(t_list) == kd_array.shape[0]
        #assert len(u_list) == kd_array.shape[0]
        t_list = np.array(t_list)#.reshape(-1,1)
        u_list = np.array(u_list)#.reshape(-1,1)
        output = np.zeros((kd_array.shape[0], kd_array.shape[1]+2))
        output[:,0] = t_list
        output[:,1] = u_list
        output[:,2:] = kd_array
        with open(kd_file, 'wb') as f:
            np.save(f, output)
        data[k]["output"][1]['shape'] = [output.shape[0], output.shape[1]]
    with open(input_json,'w', encoding='utf-8') as f:
        json.dump({'utts':data}, f, indent=4, ensure_ascii=False)
    print("Finished!.Output stored in {}".format(input_json))
        

if __name__ == '__main__':
    args = parser.parse_args()
    add_tu_list_to_kd_label(args)
