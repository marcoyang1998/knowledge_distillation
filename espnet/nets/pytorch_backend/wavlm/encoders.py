"""WavLM Encoder definition"""


import logging
import torch
import contextlib

from espnet.nets.pytorch_backend.wavlm.wavlm import WavLM, WavLMConfig
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.hubert.subsample import get_subsample_module

class WavlmEncoder(torch.nn.Module):
    def __init__(self,
                 args,
                 training=True,
                 ):
        super().__init__()

        model_dir=args.wavlm_model_dir,
        output_size=args.wavlm_output_dim,
        freeze_finetune_updates=args.wavlm_freeze_finetune_updates*args.accum_grad,
        mask_channel_prob=args.wavlm_mask_channel_prob,
        mask_prob=args.wavlm_mask_prob,
        mask_channel_length=args.wavlm_mask_channel_length,
        subsample_output=args.wavlm_subsample,
        subsample_mode=args.wavlm_subsample_mode,
        
        checkpoint = torch.load(model_dir)
        cfg = WavLMConfig(checkpoint['cfg'])
        model = WavLM(cfg)

        model.load_state_dict(checkpoint['model'])
        self.encoders = model

        if mask_channel_length != model.mask_channel_length:
            logging.warning("Overwriting mask channel length to {}. Original ckpt: {}".format(mask_channel_length, model.mask_channel_length))
            model.mask_channel_length = mask_channel_length
        if mask_channel_prob != model.mask_channel_prob:
            logging.warning("Overwriting mask channel prob to {}. Original ckpt: {}".format(mask_channel_prob, model.mask_channel_prob))
            model.mask_channel_prob = mask_channel_prob
        if mask_prob != model.mask_prob:
            logging.warning("Overwriting mask prob to {}. Original ckpt to {}".format(mask_prob, model.mask_prob))
            model.mask_prob = mask_prob

        self.encoders.feature_grad_mult = 0.0 # CNN feature extractor is frozen during finetuning!

        self.freeze_finetune_updates = freeze_finetune_updates
        self.register_buffer("num_updates", torch.LongTensor([0]))

        if subsample_output:
            self.subsample = get_subsample_module(subsample_mode, output_size)
            self.subsample_mode = subsample_mode
        else:
            self.subsample = None
            self.subsample_mode = None

        self.training = training
        self.conv_subsampling_factor = 1 # ESPnet required

    def forward(self, xs_pad, ilens):
        """Forward WavLM Encoder

        Args:
            xs_pad (torch.Tensor): padded input tensor (Batch, Tmax, Dim)
            ilens (_type_): input sequence length
        """
        mask = make_pad_mask(ilens).to(xs_pad.device)

        # check if still in freezing mode
        ft = (self.freeze_finetune_updates <= self.num_updates) and self.training

        if self.num_updates <= self.freeze_finetune_updates:
            self.num_updates += 1
        elif ft and self.num_updates == self.freeze_finetune_updates + 1:
            self.num_updates += 1
            logging.warning("Start fine-tuning Hubert parameters after {} updates!".format(self.num_updates))
        if self.num_updates%100==0:
            logging.warning("Actual batch size: {} at update: {}, finetuning transformer: {}".format(xs_pad.shape[0], self.num_updates.cpu().numpy(), ft.cpu().numpy()))
        
        apply_mask = self.training
        with torch.no_grad() if not ft else contextlib.nullcontext():
            enc_outputs = self.encoders.extract_features(
                xs_pad,
                padding_mask=mask,
                mask=apply_mask,
            )
        #xs_pad = enc_outputs["x"]
        #masks = enc_outputs["padding_mask"]
        xs_pad, masks = enc_outputs
        olens = torch.logical_not(masks).sum(dim=1)

        if self.subsample:
            if 'concat' in self.subsample_mode:
                xs_pad_1 = xs_pad[:,0:-1:2,:]
                xs_pad_2 = xs_pad[:,1::2,:]
                xs_pad = torch.cat((xs_pad_1, xs_pad_2), dim=2)
                xs_pad = self.subsample(xs_pad)
                olens = olens//2
            else:
                xs_pad = self.subsample(xs_pad.permute(0, 2, 1)).permute(0, 2, 1)
                olens = olens // 2

        return xs_pad, olens, None
