import json
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_json', type=str, help="input json file")
parser.add_argument('--dict', type=str, help="dictionary file")
parser.add_argument('--output_dir', type=str, help="output dir")
parser.add_argument('--kd_target_dir', type=str, default='', help="kd npy files dir")

def get_char_dict(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    char_dict = {}
    for line in data:
        #print(line)
        ch, id =line.split(' ')
        if id[-1] == '\n':
            id = id[:-1]
        char_dict[ch]= id
    return char_dict

def text_to_token(text, token_dict):
    token = []
    tokenid = []
    for ch in text:
        if ch in token_dict:
            token.append(ch)
            tokenid.append(token_dict[ch])
        elif ch == ' ':
            token.append('<space>')
            tokenid.append(token_dict['<space>'])
        else:
            print("Unknown character encountered: {}".format(ch))
            token.append("<unk>")
            tokenid.append(token_dict["<unk>"])
    token = ' '.join(token)
    tokenid = ' '.join(tokenid)
    return token, tokenid

def change_target(args):
    input_json = args.input_json
    dict_file = args.dict
    output_dir = args.output_dir
    kd_dir = args.kd_target_dir

    char_dict = get_char_dict(dict_file)
    odim = len(char_dict.keys())+2
    with open(input_json, 'r') as f:
        data = json.load(f)
    count  = 0
    for key in data['utts']:
        transcript = data['utts'][key]["output"][0]["text"]
        if transcript[0] == ' ':
            transcript = transcript[1:]
        if transcript[-1] == '\n':
            transcript = transcript[:-1]
        token, tokenid = text_to_token(transcript, char_dict)
        data['utts'][key]["output"][0] = {"name": "target1",
                                          "shape": [len(transcript), odim],
                                          "text": transcript,
                                          "token": token,
                                          "tokenid": tokenid}
        if kd_dir != "":
            kd_file = os.path.join(kd_dir, key+".npy")
            #spkr = '-'.join(key.split('-')[:-1])
            #kd_file = os.path.join(kd_dir, spkr, key+".npy")
            if not os.path.isfile(kd_file):
                raise ValueError("{} not found. Process aborted!".format(kd_file))
            d = np.load(kd_file)
            data['utts'][key]["output"].append({
                "name": "target2",
                "feat": kd_file,
                "shape": [d.shape[1], d.shape[2]],
                "filetype": "npy"
            })
            
        count += 1
    if kd_dir != '':
        f_name = "data_char_target_kd.json"
    else:
        f_name = "data_char_target.json"
    with open(os.path.join(output_dir, f_name),'w', encoding='utf8') as f:
        json.dump(data, f, indent=4,ensure_ascii=False)
    print("Output stored in {} with {} utterances".format(os.path.join(output_dir, f_name), count))

if __name__ == '__main__':
    args = parser.parse_args()
    change_target(args)
