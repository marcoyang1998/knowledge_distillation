import json
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Receive input')
parser.add_argument("--original_json", type=str, help="Original json file without KD path")
parser.add_argument("--kd_dir", type=str, help="Folder containing the npy file")
parser.add_argument("--include_token", default="false", type=str, help="If set to true, two targets will be used")
parser.add_argument("--output_dir", type=str, help="where to store the KD-added json")

def generate_KD_json(args):
    original_json = args.original_json
    #original_json = "../egs/librispeech/asr1/decode_w2v2/dev_other/data_w2v2.json"
    kd_dir = args.kd_dir
    #kd_dir = "/home/marcoyang/asr_project/espnet2/egs/librispeech/asr1/ctc_label/dev_other/npy_files/"
    output_dir = args.output_dir
    #output_dir = "../egs/librispeech/asr1/ctc_label/dev_other/"
    include_token = args.include_token
    #include_token = "false"
    print("Include token: ", include_token)

    with open(original_json,'r') as f:
        data = json.load(f)
    for key in data['utts']:
        kd_file = os.path.join(kd_dir, key+".npy")
        if not os.path.isfile(kd_file):
            raise ValueError("{} not found. Process aborted!".format(kd_file))
        if include_token.lower() == "true":
            data['utts'][key]["output"].append({
                        "name": "target2",
                        "feat": kd_file,
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
    with open(os.path.join(output_dir, 'data_kd_no_token.json'), 'w') as f:
        json.dump(data, f)
    print("Finish generation! Output stored in {}".format(os.path.join(output_dir, 'data_kd_no_token.json')))

if __name__ == '__main__':
    args = parser.parse_args()
    generate_KD_json(args)

