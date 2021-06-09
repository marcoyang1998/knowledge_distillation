import json
import os
import glob
import argparse
import librosa
import soundfile
import numpy

parser = argparse.ArgumentParser(description='Receive input')
parser.add_argument('--dataset_dir',type=str, help="dataset dir")
parser.add_argument('--dict_dir',type=str, help="dictionary directory")
parser.add_argument('--ext',type=str, help="audio extension")
parser.add_argument('--output_dir',type=str, help="output directory")
parser.add_argument('--odim', type=int, help="output dimension")
parser.add_argument('--kd_folder', type=str, nargs='?', default=None, help="folder storing KD training data")


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



def generate_w2v2_json_input(dataset_dir, dict_dir, ext, output_dir, odim, kd_folder):
    char_dict = get_char_dict(dict_dir)
    txt_files = glob.glob(dataset_dir + '/*/*/*/*.txt', recursive=True)
    json_dict = {}
    json_dict['utts'] = {}
    count = 0
    for txt in txt_files:
        folder_dir = '/'.join(txt.split('/')[:-1])
        with open(txt, 'r') as f:
            data = f.readlines()
        for line in data:
            count += 1
            audio_name = line.split(' ')[0]
            audio_path = os.path.join(folder_dir,audio_name+'.'+ext)
            if ext == 'flac':
                audio = soundfile.read(audio_path)[0]
                length = audio.shape[0]
            json_dict['utts'][audio_name] = {}
            json_dict['utts'][audio_name]["input"] = []
            json_dict['utts'][audio_name]["input"].append({"feat": audio_path,
                                                 "name": "input1",
                                                 "shape": [length,1],
                                                 "filetype": ext})
            transcript = line.replace(audio_name,'')
            if transcript[0] == ' ':
                transcript = transcript[1:]
            if transcript[-1] == '\n':
                transcript = transcript[:-1]
            token, tokenid = text_to_token(transcript, char_dict)
            json_dict['utts'][audio_name]["output"] = []
            json_dict['utts'][audio_name]["output"].append({"name": "target1",
                                                            "shape": [len(transcript), odim],
                                                            "text": transcript,
                                                            "token": token,
                                                            "tokenid": tokenid})
            if kd_folder:
                kd_path = os.path.join(kd_folder, audio_name+'.npy')
                if not os.path.isfile(kd_path):
                    raise ValueError("{} is not prepared!".format(kd_path))
                json_dict['utts'][audio_name]["output"].append({
                    "feat": kd_path,
                    "filetype": "npy"
                })
            json_dict['utts'][audio_name]["utt2spk"] = '-'.join(audio_name.split('-')[:-1])
    with open(os.path.join(output_dir, "data_w2v2.json"),'w') as f:
        json.dump(json_dict, f)
    print("Output stored in {} with {} utterances".format(os.path.join(output_dir, "data_w2v2.json"), count))

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


if __name__ == '__main__':
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    dict_dir = args.dict_dir
    output_dir = args.output_dir
    ext = args.ext
    odim = args.odim
    kd = args.kd_folder
    generate_w2v2_json_input(dataset_dir,dict_dir, ext, output_dir, odim, kd)