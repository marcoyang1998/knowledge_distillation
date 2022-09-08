"""Multi encoder definition for transducer models."""

from typing import List

import torch

from espnet.nets.pytorch_backend.hubert.encoder import HubertEncoder
from espnet.nets.pytorch_backend.wavlm.encoders import WavlmEncoder
from espnet.nets.pytorch_backend.rnn.encoders import Wav2VecEncoder

def get_class_from_str(etype_list: List[str]):
    """Get encoder class from a string

    Args:
        etype_list (List[str]): _description_

    Returns:
        _type_: _description_
    """
    etype_dict = {"hubert": HubertEncoder,
                  "wav2vec": Wav2VecEncoder,
                  "wavlm": WavlmEncoder}
    encoder_list = []
    for c in etype_list:
        assert c in etype_dict, f"{c} not allowed!"
        encoder_list.append(etype_dict[c])
    return encoder_list
    

class MultiEncoder(torch.nn.Module):
    """Multi encoder module for transdcuer models

    Args:
        torch (_type_): _description_
    """
    def __init__(self,
                 args,
                 idim: int,
                 enc_types: List[str],
                 combine_method: str="average",
                 training: bool=True,
                 ):
        super().__init__()
        
        encoder_cls_list = get_class_from_str(enc_types)
        self.num_encoders = len(encoder_cls_list)
        encoder_list = []
        
        for encoder_class in encoder_cls_list:
            encoder = encoder_class(args)
            encoder_list.append(encoder)
        
        encoder_dict = {f"encoder_{i}": encoder_list[i] for i in range(self.num_encoders)}
        
        self.encoder_dict = torch.nn.ModuleDict(encoder_dict)
        self.combine_method = combine_method
        self.training = training
        self.conv_subsampling_factor = 1
            
    def forward_multi_encoder(self, xs_pad, ilens):
        """Forward multi encoder

        Args:
            xs_pad (_type_): _description_
            ilens (_type_): _description_
        """
        xs_list = []
        olens_list = []
        for key in self.encoder_dict:
            xs, olens = self.encoder_dict[key](xs_pad, ilens)
            xs_list.append(xs)
            olens_list.append(olens)
        

        if self.combine_method == "average":
            xs_out = sum(xs_list) / self.num_encoders
        else:
            raise NotImplementedError()
        
        return xs_out, olens_list[0] 
    
    def forward(self, xs_pad, ilens, prev_states=None):
        
        return self.forward_multi_encoder(xs_pad, ilens)
            


