import torch

def get_subsample_module(subsample_mode, output_size):
    subsample_mode = subsample_mode
    if subsample_mode == "concat":
        subsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2*output_size, output_size),
        )
    elif subsample_mode == 'concat_relu':
        subsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2 * output_size, output_size),
            torch.nn.ReLU()
        )
    elif subsample_mode == 'concat_tanh':
        subsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2 * output_size, output_size),
            torch.nn.Tanh()
        )
    elif subsample_mode == 'avgpooling':
        subsample = torch.nn.Sequential(torch.nn.AvgPool1d(kernel_size=2, stride=2))
    else:
        raise NotImplementedError('only support: concat, concat_relu, concat_tanh, avgpooling')
        
    return subsample