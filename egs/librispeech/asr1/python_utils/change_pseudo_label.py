import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_json', type=str, help="Input json file, should be a KD json")

def change_pseudo_target(args):
    file = args.input_json

    with open(file, 'r') as f:
        data = json.load(f)

    for key in data['utts']:
        assert len(data['utts'][key]['output']) == 2, "This json should be a kd json with two targets"
        data['utts'][key]['output'][0] = {"name": 'target1',
                                        "feat": 'Pseudo',
                                        "shape": [1,1], 
                                        "filetype": "pseudo"}
    file_name = file.split('/')[-1]
    out_name = '/'.join(file.split('/')[:-1]) + '/' + file_name.replace('.json', '_pseudo.json')
    with open(out_name, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=4,ensure_ascii=False)
    print('Output stored at {}'.format(out_name))

if __name__=="__main__":
    args = parser.parse_args()
    change_pseudo_target(args)
        

