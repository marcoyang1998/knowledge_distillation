import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bad_json', type=str)
parser.add_argument('--correct_json', type=str)

def fix(args):
    bad_json = args.bad_json
    correct_json = args.correct_json
    
    
    with open(bad_json, 'r') as f:
        bad_js = json.load(f)['utts']
    with open(correct_json, 'r') as f:
        correct_js = json.load(f)['utts']
        
    for k in tqdm(bad_js):
        assert k in correct_js, k
        bad_js[k]['output'][0] = correct_js[k]['output'][0]
        
    with open(bad_json, 'w',encoding='utf-8') as f:
        json.dump({'utts': bad_js}, f, indent=4, ensure_ascii=False)
        
    print('Finished! Output stored in {} with {} utts'.format(bad_json,len(bad_js)))
    
if __name__ == '__main__':
    args = parser.parse_args()
    fix(args)
        
        
        
          
    