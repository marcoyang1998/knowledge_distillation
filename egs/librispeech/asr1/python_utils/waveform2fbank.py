import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--waveform_json", help="waveform json file")
parser.add_argument("--fbank_json", help="fbank json file")
parser.add_argument("--output_dir", help="Where to store the merged file")

def waveform2fbank(args):
    waveform_json = args.waveform_json
    fbank_json = args.fbank_json
    output_dir = args.output_dir

    with open(waveform_json,'r') as f:
        waveform_data = json.load(f)
    with open(fbank_json,'r') as f:
        fbank_data = json.load(f)
    for key in waveform_data['utts']:
        #assert key in fbank_data['utts'], print('{} not found!'.format(key))
        if key not in fbank_data['utts']:
            print('Skip {}'.format(key))
            continue
        js = waveform_data['utts'][key]
        js['input'] = fbank_data['utts'][key]['input']
        waveform_data['utts'][key] = js
    
    output_name = os.path.join(output_dir, 'data_w2f.json')
    with open(output_name, 'w', encoding='utf-8') as f:
        json.dump(waveform_data, f, indent=4, ensure_ascii=False)
    print('Output stored in {} with {} utts'.format(output_name, len(waveform_data['utts'])))

if __name__ == '__main__':
    args = parser.parse_args()
    waveform2fbank(args)
