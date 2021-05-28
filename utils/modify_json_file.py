import json
import os
import glob

def text_to_token(text, token_dict):
    token = []
    tokenid = []
    for ch in text:
        if ch == ' ':
            token.append("<space>")
            tokenid.append(token_dict["<space>"])
        elif ch.isalnum():
            token.append(ch)
            tokenid.append(token_dict[ch])
        else:
            token.append("<unk>")
            tokenid.append(token_dict["<unk>"])
    token = ' '.join(token)
    tokenid = ' '.join(tokenid)
    return token, tokenid



def generate_w2v2_json_input(dataset_dir, ext):
    char_dict = get_char_dict()
    txt_files = glob.glob(dataset_dir + '/*/*/*.txt', recursive=True)
    json_dict = {}
    json_dict['utts'] = {}
    for txt in txt_files:
        folder_dir = '/'.join(txt.split('/')[:-1])
        with open(txt, 'r') as f:
            data = f.readlines()
        for line in data:
            audio_name = line.split(' ')[0]
            audio_path = os.path.join(folder_dir,audio_name+'.flac')
            json_dict['utts'][audio_name] = {}
            json_dict['utts'][audio_name]["input"] = []
            json_dict['utts'][audio_name]["input"].append({"feat": audio_path,
                                                 "name": "input1",
                                                 "shape": [308,83],
                                                 "filetype": ext})
            transcript = line.replace(audio_name,'')
            if transcript[0] == ' ':
                transcript = transcript[1:]
            if transcript[-1] == '\n':
                transcript = transcript[:-1]
            token, tokenid = text_to_token(transcript, char_dict)
            json_dict['utts'][audio_name]["output"] = []
            json_dict['utts'][audio_name]["output"].append({"name": "target1",
                                                            "shape": [len(transcript), 30],
                                                            "text": transcript,
                                                            "token": token,
                                                            "tokenid": tokenid})
            json_dict['utts'][audio_name]["utt2spk"] = ""
    with open(dataset_dir+"/data_w2v2.json",'w') as f:
        json.dump(json_dict, f)


def get_char_dict():
    file_path = '../egs/an4/asr1/data/lang_1char/train_nodev_units.txt'
    with open(file_path, 'r') as f:
        data = f.readlines()
    char_dict = {}
    for line in data:
        ch, id =line.split(' ')
        if id[-1] == '\n':
            id = id[:-1]
        char_dict[ch]= id
    return char_dict


if __name__ == '__main__':
    #look_json()
    dataset_dir = '/home/marcoyang/Downloads/librispeech/LibriSpeech/dev-clean'
    generate_w2v2_json_input(dataset_dir,'flac')