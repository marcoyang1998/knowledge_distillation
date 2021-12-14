"""HuggingFace GPT2 language model."""

from typing import Any
from typing import List
from typing import Tuple

import logging
from typing_extensions import Required
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from espnet.nets.lm_interface import LMInterface
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.utils.cli_utils import strtobool

class GPT2LM(nn.Module, LMInterface):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--pretrained-gpt2-path", type=str, required=True
        )
        parser.add_argument(
            "--embed-unit",
            type=int,
            default=768,
            help="Number of hidden units in embedding layer",
        )
        parser.add_argument(
            "--pos-enc",
            default="none",
            choices=["sinusoidal", "none"],
            help="positional encoding",
        )
        parser.add_argument(
            "--dropout-rate", type=float, default=0.5, help="dropout probability"
        )
        return parser
    
    def __init__(self, n_vocab, args):
        """Initialize a GPT2 language model in espnet

        Args:
            n_vocab ([type]): [description]
            args ([type]): [description]
        """
        nn.Module.__init__(self)
        
        emb_dropout_rate = getattr(args, "emb_dropout_rate", 0.0)
        pretrained_gpt2_path = args.pretrained_gpt2_path
        
        
        if args.pos_enc == "sinusoidal":
            pos_enc_class = PositionalEncoding
        elif args.pos_enc == "none":

            def pos_enc_class(*args, **kwargs):
                return nn.Sequential()  # indentity
        else:
            raise ValueError(f"unknown pos-enc option: {args.pos_enc}")
        
        self.embed = nn.Embedding(n_vocab, args.embed_unit)
        if emb_dropout_rate == 0.0:
            self.embed_drop = None
        else:
            self.embed_drop = nn.Dropout(emb_dropout_rate)
        
        pretrained_model = GPT2LMHeadModel.from_pretrained(pretrained_gpt2_path)
        pretrained_model.resize_token_embeddings(n_vocab)
        self.encoder = pretrained_model.transformer
        self.decoder = pretrained_model.lm_head
        #self.pretrained_model = pretrained_model
        
    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        #print(x.shape)
        xm = x != 0
        if self.embed_drop is not None:
            emb = self.embed_drop(self.embed(x))
        else:
            emb = self.embed(x)
        #transformer_outputs = self.encoder(inputs_embeds=emb)
        transformer_outputs = self.encoder(x)
        hidden_states = transformer_outputs[0]
        lm_logits = self.decoder(hidden_states)
        
        shift_logits = lm_logits[..., :, :].contiguous()
        shift_labels = t[..., :].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        #loss = F.cross_entropy(lm_logits.view(-1, lm_logits.shape[-1]), t.view(-1), reduction="none")
        
        mask = xm.to(dtype=loss.dtype)
        logp = loss * mask.view(-1)
        logp = logp.sum()
        count = mask.sum()
        
        return logp / count, logp, count

    def score(self, y: torch.Tensor, state: Any, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return super().score(y, state, x)
    
    def batch_score(self):
        return 