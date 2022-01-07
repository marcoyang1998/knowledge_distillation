import json
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='To create KD-mtl json, the input json should be a normal ctc-json file. To create a KD only json, use and input json which has the right input feature')
parser.add_argument("--original_json", type=str, help="Original json file without KD path")
parser.add_argument("--kd_dir", type=str, help="Folder containing the npy file")
parser.add_argument("--include_token", default="false", type=str, help="If set to true, two targets will be used")
parser.add_argument("--output_dir", type=str, help="where to store the KD-added json")

def generate_KD_json(args):
    original_json = args.original_json
    kd_dir = args.kd_dir
    output_dir = args.output_dir
    include_token = args.include_token
    print("Include token: ", include_token)

    with open(original_json,'r') as f:
        data = json.load(f)
    count = 0
    new_js = {}
    for key in data['utts']:
        #kd_file = os.path.join(kd_dir, key+".npy")
        spkr = '-'.join(key.split('-')[:-1])
        kd_file = os.path.join(kd_dir, key.split('-')[0], spkr, key+".npy")
        if not os.path.isfile(kd_file):
            print(("{} not found.".format(kd_file)))
            count += 1
            continue
            #raise ValueError("{} not found. Process aborted!".format(kd_file))
        if include_token.lower() == "true":
            d = np.load(kd_file)
            data['utts'][key]["output"].append({
                        "name": "target2",
                        "feat": kd_file,
                        "shape": [d.shape[1], d.shape[2]],
                        "filetype": "npy"
                    })
        else:
            d = np.load(kd_file)
            data['utts'][key]["output"] = [{
                "name": "target1",
                "feat": kd_file,
                "shape": [d.shape[1], d.shape[2]],
                "filetype": "npy"
            }]
        new_js[key] = data['utts'][key]
    if include_token.lower() == "true":
        name = 'data_kd_with_token.json'
    else:
        name = 'data_kd_no_token.json'
    new_js = {'utts':new_js}
    with open(os.path.join(output_dir, name), 'w', encoding='utf8') as f:
        json.dump(new_js, f, indent=4,ensure_ascii=False)
    print("Finish generation! Output stored in {}".format(os.path.join(output_dir, name)))
    print("{}/{} utterances not found!".format(count, len(new_js['utts'])))

if __name__ == '__main__':
    args = parser.parse_args()
    generate_KD_json(args)
