from argparse import ArgumentParser
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--input_txt', type=str)

def check_length(args):
    input_txt = args.input_txt
    new_txt = []
    count = 0
    with open(input_txt, 'r') as f:
        data = f.readlines()
    for line in tqdm(data):
        line_lst = line.strip().split()
        if len(line_lst) > 1000:
            count += 1
            continue
        new_txt.append(line)
    output_name = input_txt.replace('.txt','_shortened.txt')
    with open(output_name, 'w') as f:
        f.writelines(new_txt)
            
    print(f'Count: {count}')
    
if __name__=="__main__":
    args = parser.parse_args()
    check_length(args)
    