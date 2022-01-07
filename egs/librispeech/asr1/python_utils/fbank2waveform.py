import json
import os
import argparse
from posixpath import join
import soundfile

parser = argparse.ArgumentParser()
parser.add_argument('--input_json', type=str)
parser.add_argument('--file_list', type=str, help='train-clean-100 or ...')
parser.add_argument('--dict_dir',type=str, help="dictionary directory")
parser.add_argument('--output_folder', type=str)
parser.add_argument('--ext', default='flac', type=str)

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
        elif ch == '\'':
            token.append('\'')
            tokenid.append(token_dict['\''])
        else:
            print("Unknown character encountered: {}".format(ch))
            token.append("<unk>")
            tokenid.append(token_dict["<unk>"])
    token = ' '.join(token)
    tokenid = ' '.join(tokenid)
    return token, tokenid

def convert(args):
    in_json = args.input_json
    file_list = args.file_list
    ext = args.ext
    output_folder = args.output_folder
    dict_dir = args.dict_dir
    if dict_dir:
        char_dict = get_char_dict(dict_dir)

    with open(in_json,'r') as f:
        data = json.load(f)

    with open(file_list, 'r') as f:
        list_json = json.load(f)

    for key in data['utts']:
        assert key in list_json, "{} cannot be found".format(key)
        assert os.path.isfile(list_json[key]), "{} cannoot be found".format(list_json[key])
        if ext == 'flac':
            audio = soundfile.read(list_json[key])[0]
            length = audio.shape[0]
            del audio
        data['utts'][key]['input'][0] = {"feat": list_json[key],
                                                 "name": "input1",
                                                 "shape": [length,1],
                                                 "filetype": ext}
        if dict_dir:
            transcript = data['utts'][key]['output'][0]['text']
            if transcript[0] == ' ':
                transcript = transcript[1:]
            if transcript[-1] == '\n':
                transcript = transcript[:-1]
            token, tokenid = text_to_token(transcript, char_dict)
            data['utts'][key]['output'][0] = {"name": "target1",
                                            "shape": [len(transcript), 31],
                                            "text": transcript,
                                            "token": token,
                                            "tokenid": tokenid}

    output_name = os.path.join(output_folder, 'data.json')
    with open(output_name, 'w',encoding='utf8') as f:
        json.dump(data, f, indent=4,ensure_ascii=False)
    print("Output stored in {}".format(output_name))

if __name__ == '__main__':
    args = parser.parse_args()
    convert(args)
