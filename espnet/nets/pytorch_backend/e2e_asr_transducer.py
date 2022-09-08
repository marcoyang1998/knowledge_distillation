"""Transducer speech recognition model (pytorch)."""

from argparse import Namespace
from collections import Counter
from dataclasses import asdict
import logging
import math
from typing import List
import numpy

import chainer
import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transducer.arguments import (
    add_encoder_general_arguments,  # noqa: H301
    add_rnn_encoder_arguments,  # noqa: H301
    add_custom_encoder_arguments,  # noqa: H301
    add_decoder_general_arguments,  # noqa: H301
    add_rnn_decoder_arguments,  # noqa: H301
    add_custom_decoder_arguments,  # noqa: H301
    add_custom_training_arguments,  # noqa: H301
    add_transducer_arguments,  # noqa: H301
    add_auxiliary_task_arguments,  # noqa: H301
)
from espnet.nets.pytorch_backend.wav2vec2.argument import add_arguments_w2v2_common
from espnet.nets.pytorch_backend.hubert.argument import add_arguments_hubert_common
from espnet.nets.pytorch_backend.wavlm.argument import add_arguments_wavlm_common
from espnet.nets.pytorch_backend.transducer.auxiliary_task import AuxiliaryTask
from espnet.nets.pytorch_backend.transducer.custom_decoder import CustomDecoder
from espnet.nets.pytorch_backend.transducer.custom_encoder import CustomEncoder
from espnet.nets.pytorch_backend.transducer.error_calculator import ErrorCalculator
from espnet.nets.pytorch_backend.transducer.initializer import initializer
from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork
from espnet.nets.pytorch_backend.transducer.loss import TransLoss
from espnet.nets.pytorch_backend.transducer.rnn_decoder import DecoderRNNT
from espnet.nets.pytorch_backend.transducer.rnn_encoder import encoder_for
from espnet.nets.pytorch_backend.hubert.encoder import HubertEncoder
from espnet.nets.pytorch_backend.wavlm.encoders import WavlmEncoder
from espnet.nets.pytorch_backend.transducer.utils import prepare_loss_inputs
from espnet.nets.pytorch_backend.transducer.utils import valid_aux_task_layer_list
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.utils.fill_missing_args import fill_missing_args
from espnet.nets.pytorch_backend.nets_utils import pad_list


class Reporter(chainer.Chain):
    """A chainer reporter wrapper for transducer models."""

    def report(
        self,
        loss,
        loss_trans,
        loss_ctc,
        loss_lm,
        loss_aux_trans,
        loss_aux_symm_kl,
        cer,
        wer,
    ):
        """Instantiate reporter attributes."""
        chainer.reporter.report({"loss": loss}, self)
        chainer.reporter.report({"loss_trans": loss_trans}, self)
        chainer.reporter.report({"loss_ctc": loss_ctc}, self)
        chainer.reporter.report({"loss_lm": loss_lm}, self)
        chainer.reporter.report({"loss_aux_trans": loss_aux_trans}, self)
        chainer.reporter.report({"loss_aux_symm_kl": loss_aux_symm_kl}, self)
        chainer.reporter.report({"cer": cer}, self)
        chainer.reporter.report({"wer": wer}, self)

        logging.info("loss:" + str(loss))

    def report_kd(
        self,
        loss,
        loss_trans,
        loss_ctc,
        loss_lm,
        loss_aux_trans,
        loss_aux_symm_kl,
        loss_kd,
        cer,
        wer,
    ):
        """Instantiate reporter attributes."""
        chainer.reporter.report({"loss": loss}, self)
        chainer.reporter.report({"loss_trans": loss_trans}, self)
        chainer.reporter.report({"loss_ctc": loss_ctc}, self)
        chainer.reporter.report({"loss_lm": loss_lm}, self)
        chainer.reporter.report({"loss_aux_trans": loss_aux_trans}, self)
        chainer.reporter.report({"loss_aux_symm_kl": loss_aux_symm_kl}, self)
        chainer.reporter.report({"loss_kd": loss_kd}, self)
        chainer.reporter.report({"cer": cer}, self)
        chainer.reporter.report({"wer": wer}, self)

        logging.info("loss:" + str(loss))


class E2E(ASRInterface, torch.nn.Module):
    """E2E module for transducer models.
    Args:
        idim (int): dimension of inputs
        odim (int): dimension of outputs
        args (Namespace): argument Namespace containing options
        ignore_id (int): padding symbol id
        blank_id (int): blank symbol id
    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments for transducer model."""
        E2E.encoder_add_general_arguments(parser)
        E2E.encoder_add_rnn_arguments(parser)
        E2E.encoder_add_custom_arguments(parser)
        E2E.encoder_add_w2v2_arguments(parser)
        E2E.encoder_add_hubert_arguments(parser)
        E2E.encoder_add_wavlm_arguments(parser)


        E2E.decoder_add_general_arguments(parser)
        E2E.decoder_add_rnn_arguments(parser)
        E2E.decoder_add_custom_arguments(parser)

        E2E.training_add_custom_arguments(parser)
        E2E.transducer_add_arguments(parser)
        E2E.auxiliary_task_add_arguments(parser)

        return parser

    @staticmethod
    def encoder_add_general_arguments(parser):
        """Add general arguments for encoder."""
        group = parser.add_argument_group("Encoder general arguments")
        group = add_encoder_general_arguments(group)

        return parser

    @staticmethod
    def encoder_add_rnn_arguments(parser):
        """Add arguments for RNN encoder."""
        group = parser.add_argument_group("RNN encoder arguments")
        group = add_rnn_encoder_arguments(group)

        return parser

    @staticmethod
    def encoder_add_w2v2_arguments(parser):
        group = parser.add_argument_group("W2V2 encoder arguments")
        group = add_arguments_w2v2_common(group)

        return parser
    
    @staticmethod
    def encoder_add_hubert_arguments(parser):
        group = parser.add_argument_group("Hubert encoder arguments")
        group = add_arguments_hubert_common(group)

        return parser

    @staticmethod
    def encoder_add_wavlm_arguments(parser):
        group = parser.add_argument_group("WavLM encoder arguments")
        group = add_arguments_wavlm_common(group)

        return parser

    @staticmethod
    def encoder_add_custom_arguments(parser):
        """Add arguments for Custom encoder."""
        group = parser.add_argument_group("Custom encoder arguments")
        group = add_custom_encoder_arguments(group)

        return parser

    @staticmethod
    def decoder_add_general_arguments(parser):
        """Add general arguments for decoder."""
        group = parser.add_argument_group("Decoder general arguments")
        group = add_decoder_general_arguments(group)

        return parser

    @staticmethod
    def decoder_add_rnn_arguments(parser):
        """Add arguments for RNN decoder."""
        group = parser.add_argument_group("RNN decoder arguments")
        group = add_rnn_decoder_arguments(group)

        return parser

    @staticmethod
    def decoder_add_custom_arguments(parser):
        """Add arguments for Custom decoder."""
        group = parser.add_argument_group("Custom decoder arguments")
        group = add_custom_decoder_arguments(group)

        return parser

    @staticmethod
    def training_add_custom_arguments(parser):
        """Add arguments for Custom architecture training."""
        group = parser.add_argument_group("Training arguments for custom archictecture")
        group = add_custom_training_arguments(group)

        return parser

    @staticmethod
    def transducer_add_arguments(parser):
        """Add arguments for transducer model."""
        group = parser.add_argument_group("Transducer model arguments")
        group = add_transducer_arguments(group)

        return parser

    @staticmethod
    def auxiliary_task_add_arguments(parser):
        """Add arguments for auxiliary task."""
        group = parser.add_argument_group("Auxiliary task arguments")
        group = add_auxiliary_task_arguments(group)

        return parser

    @property
    def attention_plot_class(self):
        """Get attention plot class."""
        return PlotAttentionReport

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        if self.etype == "custom":
            return self.encoder.conv_subsampling_factor * int(
                numpy.prod(self.subsample)
            )
        else:
            return self.enc.conv_subsampling_factor * int(numpy.prod(self.subsample))

    def __init__(self, idim, odim, args, blank_id=0, training=True):
        """Construct an E2E object for transducer model."""
        torch.nn.Module.__init__(self)

        args = fill_missing_args(args, self.add_arguments)

        self.is_rnnt = True
        self.transducer_weight = args.transducer_weight

        self.ignore_id = getattr(args, "ignore_id", -512)
        logging.info("Ignore_id: {}".format(self.ignore_id))

        self.use_aux_task = (
            True if (args.aux_task_type is not None and training) else False
        )

        self.use_aux_ctc = args.aux_ctc and training
        self.aux_ctc_weight = args.aux_ctc_weight

        self.use_aux_cross_entropy = args.aux_cross_entropy and training
        self.aux_cross_entropy_weight = args.aux_cross_entropy_weight

        if self.use_aux_task:
            n_layers = (
                (len(args.enc_block_arch) * args.enc_block_repeat - 1)
                if args.enc_block_arch is not None
                else (args.elayers - 1)
            )

            aux_task_layer_list = valid_aux_task_layer_list(
                args.aux_task_layer_list,
                n_layers,
            )
        else:
            aux_task_layer_list = []

        if "custom" in args.etype:
            if args.enc_block_arch is None:
                raise ValueError(
                    "When specifying custom encoder type, --enc-block-arch"
                    "should also be specified in training config. See"
                    "egs/vivos/asr1/conf/transducer/train_*.yaml for more info."
                )

            self.subsample = get_subsample(args, mode="asr", arch="transformer")

            self.encoder = CustomEncoder(
                idim,
                args.enc_block_arch,
                input_layer=args.custom_enc_input_layer,
                repeat_block=args.enc_block_repeat,
                self_attn_type=args.custom_enc_self_attn_type,
                positional_encoding_type=args.custom_enc_positional_encoding_type,
                positionwise_activation_type=args.custom_enc_pw_activation_type,
                conv_mod_activation_type=args.custom_enc_conv_mod_activation_type,
                aux_task_layer_list=aux_task_layer_list,
                streaming=args.streaming,
                encoder_projection=args.encoder_projection,
            )
            encoder_out = self.encoder.enc_out if args.encoder_projection == 0 else args.encoder_projection

            self.most_dom_list = args.enc_block_arch[:]
        else:
            self.subsample = get_subsample(args, mode="asr", arch="rnn-t")

            if args.etype == "hubert":
                self.enc = HubertEncoder(
                    model_dir=args.hubert_model_dir,
                    output_size=args.hubert_output_dim,
                    freeze_finetune_updates=args.hubert_freeze_finetune_updates*args.accum_grad,
                    mask_channel_prob=args.hubert_mask_channel_prob,
                    mask_prob=args.hubert_mask_prob,
                    mask_channel_length=args.hubert_mask_channel_length,
                    subsample_output=args.hubert_subsample,
                    subsample_mode=args.hubert_subsample_mode,
                    training=training,
                )
                encoder_out = args.hubert_output_dim
            elif args.etype == "wavlm":
                self.enc = WavlmEncoder(
                    model_dir=args.wavlm_model_dir,
                    output_size=args.wavlm_output_dim,
                    freeze_finetune_updates=args.wavlm_freeze_finetune_updates*args.accum_grad,
                    mask_channel_prob=args.wavlm_mask_channel_prob,
                    mask_prob=args.wavlm_mask_prob,
                    mask_channel_length=args.wavlm_mask_channel_length,
                    subsample_output=args.wavlm_subsample,
                    subsample_mode=args.wavlm_subsample_mode,
                    training=training,
                )
                encoder_out = args.wavlm_output_dim
            else:
                self.enc = encoder_for(
                    args,
                    idim,
                    self.subsample,
                    aux_task_layer_list=aux_task_layer_list,
                )
                encoder_out = args.eprojs
        if args.modify_first_block and args.streaming and args.first_block_future_context > 0: # modify the first encoder block to allow for furture context, only used when streaming
            n_head = args.enc_block_arch[0]['heads']
            n_feat = args.enc_block_arch[0]['d_hidden']
            att_dropout = args.enc_block_arch[0]['att-dropout-rate']
            self.encoder.encoders[0].self_attn = RelPositionMultiHeadedAttention(n_head=n_head, 
                                                                                 n_feat=n_feat, 
                                                                                 dropout_rate=att_dropout,
                                                                                 zero_triu=args.streaming, 
                                                                                 future_context=args.first_block_future_context)

        if "custom" in args.dtype:
            if args.dec_block_arch is None:
                raise ValueError(
                    "When specifying custom decoder type, --dec-block-arch"
                    "should also be specified in training config. See"
                    "egs/vivos/asr1/conf/transducer/train_*.yaml for more info."
                )

            self.decoder = CustomDecoder(
                odim,
                args.dec_block_arch,
                input_layer=args.custom_dec_input_layer,
                repeat_block=args.dec_block_repeat,
                positionwise_activation_type=args.custom_dec_pw_activation_type,
                dropout_rate_embed=args.dropout_rate_embed_decoder,
            )
            decoder_out = self.decoder.dunits

            if "custom" in args.etype:
                self.most_dom_list += args.dec_block_arch[:]
            else:
                self.most_dom_list = args.dec_block_arch[:]
        else:
            self.dec = DecoderRNNT(
                odim,
                args.dtype,
                args.dlayers,
                args.dunits,
                blank_id,
                args.dec_embed_dim,
                args.dropout_rate_decoder,
                args.dropout_rate_embed_decoder,
                ignore_id=self.ignore_id,
                dproj_dim=args.dproj_dim
            )
        decoder_out = args.dunits if args.dproj_dim == 0 else args.dproj_dim
        self.decoder_out = decoder_out
        
        self.joint_network = JointNetwork(
            odim, encoder_out, decoder_out, args.joint_dim, args.joint_activation_type
        )
        
        if args.dproj_dim > 0:
            logging.info(self.dec.dproj)
            logging.info(self.joint_network.lin_dec)
            logging.warning("Add projection layer after RNN predictor")
            
        if hasattr(self, "most_dom_list"):
            self.most_dom_dim = sorted(
                Counter(
                    d["d_hidden"] for d in self.most_dom_list if "d_hidden" in d
                ).most_common(),
                key=lambda x: x[0],
                reverse=True,
            )[0][0]
        else:
            self.most_dom_dim = args.dunits/2
        self.etype = args.etype
        self.dtype = args.dtype

        self.sos = odim - 1
        self.eos = odim - 1
        self.blank_id = blank_id

        self.space = args.sym_space
        self.blank = args.sym_blank

        self.odim = odim
        self.do_kd = args.do_knowledge_distillation

        self.reporter = Reporter()

        self.error_calculator = None

        self.default_parameters(args)

        self.trans_loss_reduction = args.trans_loss_reduction
        self.criterion = TransLoss(args.trans_type, self.blank_id, reduction=self.trans_loss_reduction)
        if training:
            self.trans_loss_reduction = args.trans_loss_reduction
            self.criterion = TransLoss(args.trans_type, self.blank_id, reduction=self.trans_loss_reduction)

            decoder = self.decoder if self.dtype == "custom" else self.dec

            if args.report_cer or args.report_wer:
                self.error_calculator = ErrorCalculator(
                    decoder,
                    self.joint_network,
                    args.char_list,
                    args.sym_space,
                    args.sym_blank,
                    args.report_cer,
                    args.report_wer,
                    ignore_id=self.ignore_id
                )

            if self.use_aux_task:
                self.auxiliary_task = AuxiliaryTask(
                    decoder,
                    self.joint_network,
                    self.criterion,
                    args.aux_task_type,
                    args.aux_task_weight,
                    encoder_out,
                    args.joint_dim,
                )

            if self.use_aux_ctc:
                self.aux_ctc = ctc_for(
                    Namespace(
                        num_encs=1,
                        eprojs=encoder_out,
                        dropout_rate=args.aux_ctc_dropout_rate,
                        ctc_type="warpctc",
                    ),
                    odim,
                )

            if self.use_aux_cross_entropy:
                self.aux_decoder_output = torch.nn.Linear(decoder_out, odim)

                self.aux_cross_entropy = LabelSmoothingLoss(
                    odim, self.ignore_id, args.aux_cross_entropy_smoothing
                )
            
            self.use_ILM_gt_loss = args.ILM_gt_loss
            self.ILM_gt_loss_factor = args.ILM_gt_loss_factor
                
            if self.do_kd:
                self.kd_mtl_factor = args.kd_mtl_factor
                self.kd_temperature = args.kd_temperature
                self.kd_mode = args.transducer_kd_mode
                self.kd_prob_label = args.kd_prob_label
                if self.kd_mode == "one_best_path":
                    self.kd_loss = self.kd_one_best_loss
                elif self.kd_mode == "reduced_lattice":
                    self.kd_loss = self.kd_reduced_lattice_loss
                elif self.kd_mode == "shifted_one_best_path":
                    self.kd_loss = self.kd_shifted_one_best_loss
                    self.shift_step = args.shift_step
                elif self.kd_mode == "window_shifted_one_best_path":
                    self.kd_loss = self.kd_window_shifted_one_best_loss
                    #self.kd_loss = self.shifted_one_best_path_loss
                    self.shift_step = args.shift_step
                else:
                    raise NotImplementedError("Not implemented {}".format(self.kd_mode))
                
                if self.kd_mtl_factor == 0:
                    logging.warning('Lattice based CXE kd loss will not be calculated')
                else:
                    logging.warning("Distillation mode: {}, kd factor: {}".format(self.kd_mode, self.kd_mtl_factor))
                
                self.kd_ILM_loss_factor = args.kd_ILM_loss_factor
                self.kd_ILM_teacher_weight = args.kd_ILM_teacher_weight
                if self.kd_ILM_loss_factor == 0:
                    logging.warning('CXE ILM kd loss will not be calculated!')
                else:
                    logging.warning(f'CXE ILM kd loss will be used with a factor of {self.kd_ILM_loss_factor}, Teacher LM weight is {self.kd_ILM_teacher_weight}!')
                    
                self.kd_loss_reduction = args.kd_loss_reduction
                logging.warning('KD loss reduction mode is set to {}'.format(self.kd_loss_reduction))
                    
        self.loss = None
        self.rnnlm = None
        self.dec_feature_loss_factor = args.dec_feature_loss_factor
        self.use_dec_feature_loss = args.use_dec_feature_loss and self.dec_feature_loss_factor > 0
        
        if args.streaming:
            logging.warning("This is a streaming transducer!")
        self.use_dproj = False
            
    def add_dproj_layer(self, args, device):
        
        self.use_dproj = True

    def update_joiner(self, args, device):
        
        self.joint_network = JointNetwork(
            self.odim, args.encoder_projection, self.decoder_out, args.joint_dim, args.joint_activation_type
        )
        initializer(self, args, name_list=['joint_network'])
        self.joint_network.to(device)
        if args.report_cer or args.report_wer:
            self.error_calculator = ErrorCalculator(
                self.decoder if self.dtype == "custom" else self.dec,
                self.joint_network,
                args.char_list,
                args.sym_space,
                args.sym_blank,
                args.report_cer,
                args.report_wer,
                args.ignore_id
            )
                       
        logging.warning(self.joint_network)
        
    def default_parameters(self, args):
        """Initialize/reset parameters for transducer.
        Args:
            args (Namespace): argument Namespace containing options
        """
        initializer(self, args)

    def kd_one_best_loss(self, z, pred_len, target_len, enc_T, ys_pad, ys_pad_kd, reduction='mean'):
        # 0: t_list, 1: u_list， 2： y_seq_with_blank
        def CXE(target, predicted):
            return -(target * predicted.log_softmax(-1)).sum()

        kd_mask = ys_pad_kd[:,:,0] != self.ignore_id
        bs = z.shape[0]
        #ys = [y[y != self.ignore_id] for y in ys_pad]
        t_list = [y[:, 0][y[:, 0] != self.ignore_id] for y in ys_pad_kd]
        u_list = [y[:, 1][y[:, 1] != self.ignore_id] for y in ys_pad_kd]
        kd_seq = [y[:, 2][y[:, 2] != self.ignore_id] for y in ys_pad_kd]
        kd_seq_no_blank = [seq[seq > 0] for seq in kd_seq]
        #for i in range(bs): assert torch.equal(ys[i], kd_seq_no_blank[i]), "ys: {}, kd: {}".format(ys[i],                                                                                                   kd_seq_no_blank[i])
        kd_seq_no_blank_len = [seq.size(0) for seq in kd_seq_no_blank]
        min_T = [min(enc_T[i], torch.max(t_list[i]) + 1) for i in range(bs)]
        t_mask = [l <= min_T[i] -1 for i, l in enumerate(t_list)]
        u_mask = [l <= target_len[i] for i, l in enumerate(u_list)]
        mask = [t_mask[i]*u_mask[i] for i in range(bs)]
        # for i in range(bs): assert target_len[i] == kd_seq_no_blank_len[i], "target: {}, kd: {}".format(ys[i], kd_seq_no_blank[i])

        ys_kd = [y[y != self.ignore_id].view(-1, self.odim) for i, y in enumerate(ys_pad_kd[:, :, 3:])]
        ys_kd = [y[mask[i]] for i, y in enumerate(ys_kd)]
        u_list = [l[mask[i]] for i, l in enumerate(u_list)]
        t_list = [l[mask[i]] for i, l in enumerate(t_list)]
        #assert max([max(l) for l in t_list]) < z.shape[1], print([max(l) for l in t_list], z.shape)
        #assert max([max(l) for l in u_list]) < z.shape[2], print([max(l) for l in u_list], z.shape)

        logits = [lattice[t_list[i].long(), u_list[i].long(), :] for i, lattice in enumerate(z)]
        if reduction == 'none':
            kd_loss_list = []
            for i in range(bs):
                kd_loss_list.append(CXE(torch.softmax(ys_kd[i] / self.kd_temperature, dim=-1), 
                                        logits[i] / self.kd_temperature))
            return kd_loss_list
        else:
            ys_kd = torch.cat(ys_kd, dim=0)
            logits = torch.cat(logits, dim=0)
            count = kd_mask.sum()
            if self.kd_prob_label:
                kd_loss = CXE(ys_kd, logits)
            else:
                kd_loss = CXE(torch.softmax(ys_kd / self.kd_temperature, dim=-1), logits / self.kd_temperature)
            if self.kd_loss_reduction == "node":
                return kd_loss/count
            else:
                return kd_loss/bs

    def kd_reduced_lattice_loss(self, z, pred_len, target_len, enc_T, ys_pad, ys_pad_kd):
        def reduced_CXE(target, predicted):
            # target is already probability
            # predicted is already probability
            return -(target * torch.log(predicted)).sum()

        bs = z.shape[0]
        ys = [y[y != self.ignore_id] for y in ys_pad]
        ys_kd = [y[y != self.ignore_id].view(-1,target_len[i], 3) for i,y in enumerate(ys_pad_kd)]
        min_len_T = [min(ys_kd[i].size(0), enc_T[i]) for i in range(bs)]
        min_len_U = [min(ys[i].size(0), ys_kd[i].size(1)) for i in range(bs)]
        ys_kd = [ys_kd[i][:min_len_T[i], :min_len_U[i],:] for i in range(bs)]
        pr = z.softmax(dim=-1)
        pr = [pr[i, :min_len_T[i], :min_len_U[i],:] for i in range(bs)]

        blank = torch.cat([lattice[:,:,0].transpose(0,1).reshape(-1) for lattice in pr]).view(-1,1)
        correct = []
        #rest = []
        for b in range(bs):
            correct.append(torch.cat([pr[b][:,i,ys[b][i]] for i in range(ys[b].size(0))]))
            #rest.append(torch.cat([pr[b][:,i,numpy.delete(numpy.arange(258), [0, ys[b][i].cpu()])].sum(-1) for i in range(ys[b].size(0))]))
        correct = torch.cat(correct).view(-1,1)
        #rest = torch.cat(rest).view(-1,1)
        ys_kd = torch.cat([torch.swapaxes(y,0,1).reshape(-1,3) for y in ys_kd], dim=0)
        #reduced_lattice = torch.cat((blank, correct, rest), dim=-1)
        reduced_lattice = torch.cat((blank, correct, 1.0 - correct - blank + 1e-7), dim=-1)
        if reduced_lattice.min() < 0:
            print(reduced_lattice.min())

        return reduced_CXE(ys_kd, reduced_lattice)/bs

    def kd_shifted_one_best_loss(self, z, pred_len, target_len, enc_T, ys_pad, ys_pad_kd):
        def CXE(target, predicted):
            return -(target * predicted.log_softmax(-1)).sum()

        bs = z.shape[0]
        # ys = [y[y != self.ignore_id] for y in ys_pad]
        t_list = [y[:, 0][y[:, 0] != self.ignore_id] + self.shift_step for y in ys_pad_kd]
        u_list = [y[:, 1][y[:, 1] != self.ignore_id] for y in ys_pad_kd]
        kd_seq = [y[:, 2][y[:, 2] != self.ignore_id] for y in ys_pad_kd]
        kd_seq_no_blank = [seq[seq > 0] for seq in kd_seq]
        # for i in range(bs): assert torch.equal(ys[i], kd_seq_no_blank[i]), "ys: {}, kd: {}".format(ys[i],                                                                                                   kd_seq_no_blank[i])
        #kd_seq_no_blank_len = [seq.size(0) for seq in kd_seq_no_blank]
        min_T = [min(enc_T[i], torch.max(t_list[i]) + 1) for i in range(bs)]
        t_mask = [l <= min_T[i] - 1 for i, l in enumerate(t_list)]
        u_mask = [l <= target_len[i] for i, l in enumerate(u_list)]
        mask = [t_mask[i] * u_mask[i] for i in range(bs)]
        # for i in range(bs): assert target_len[i] == kd_seq_no_blank_len[i], "target: {}, kd: {}".format(ys[i], kd_seq_no_blank[i])

        ys_kd = [y[y != self.ignore_id].view(-1, self.odim) for i, y in enumerate(ys_pad_kd[:, :, 3:])]
        ys_kd = [y[mask[i]] for i, y in enumerate(ys_kd)]
        u_list = [l[mask[i]] for i, l in enumerate(u_list)]
        t_list = [l[mask[i]] for i, l in enumerate(t_list)]

        logits = [lattice[t_list[i].long(), u_list[i].long(), :] for i, lattice in enumerate(z)]
        ys_kd = torch.cat(ys_kd, dim=0)
        logits = torch.cat(logits, dim=0)
        kd_loss = CXE(torch.softmax(ys_kd / self.kd_temperature, dim=-1), logits / self.kd_temperature)

        return kd_loss/bs

    def kd_window_shifted_one_best_loss(self, z, pred_len, target_len, enc_T, ys_pad, ys_pad_kd):
        def CXE(target, predicted):
            return -(target * predicted.log_softmax(-1)).sum()
        def CEX_no_sum(target, predicted):
            return -(target * predicted.log_softmax(-1))

        bs = z.shape[0]
        # ys = [y[y != self.ignore_id] for y in ys_pad]
        t_list = [y[:, 0][y[:, 0] != self.ignore_id] + self.shift_step for y in ys_pad_kd]
        u_list = [y[:, 1][y[:, 1] != self.ignore_id] for y in ys_pad_kd]
        kd_seq = [y[:, 2][y[:, 2] != self.ignore_id] for y in ys_pad_kd]
        kd_seq_no_blank = [seq[seq > 0] for seq in kd_seq]
        # for i in range(bs): assert torch.equal(ys[i], kd_seq_no_blank[i]), "ys: {}, kd: {}".format(ys[i],                                                                                                   kd_seq_no_blank[i])
        # kd_seq_no_blank_len = [seq.size(0) for seq in kd_seq_no_blank]
        min_T = [min(enc_T[i], torch.max(t_list[i]) + 1) for i in range(bs)]
        t_mask = [l <= min_T[i] - 1 for i, l in enumerate(t_list)]
        u_mask = [l <= target_len[i] for i, l in enumerate(u_list)]
        mask = [t_mask[i] * u_mask[i] for i in range(bs)]
        # for i in range(bs): assert target_len[i] == kd_seq_no_blank_len[i], "target: {}, kd: {}".format(ys[i], kd_seq_no_blank[i])

        ys_kd = [y[y != self.ignore_id].view(-1, self.odim) for i, y in enumerate(ys_pad_kd[:, :, 3:])]
        ys_kd = [y[mask[i]] for i, y in enumerate(ys_kd)]
        u_list = [l[mask[i]] for i, l in enumerate(u_list)]
        t_list = [l[mask[i]] for i, l in enumerate(t_list)]
        t_list_left = [l - 2 for l in t_list]
        t_list_middle = [l - 1 for l in t_list]

        logits = [lattice[t_list[i].long(), u_list[i].long(), :] for i, lattice in enumerate(z)]
        logits_left = [lattice[t_list_left[i].long(), u_list[i].long(), :] for i, lattice in enumerate(z)]
        logits_middle = [lattice[t_list_middle[i].long(), u_list[i].long(), :] for i, lattice in enumerate(z)]
        full_logits = [logits_left, logits_middle, logits]

        ys_kd = torch.cat(ys_kd, dim=0)
        kd_loss_list = []
        for l in full_logits:
            l = torch.cat(l, dim=0)
            kd_loss_list.append(CEX_no_sum(torch.softmax(ys_kd / self.kd_temperature, dim=-1), l / self.kd_temperature).sum(1))

        kd_loss_list = torch.cat(kd_loss_list).reshape(-1, 3)
        kd_loss = torch.min(kd_loss_list, dim=1)[0].sum()
        #kd_loss = CEX_no_sum(torch.softmax(ys_kd / self.kd_temperature, dim=-1), logits / self.kd_temperature).sum(1)
        #kd_loss = CXE(torch.softmax(ys_kd / self.kd_temperature, dim=-1), logits / self.kd_temperature)

        return kd_loss / bs

    def kd_ILM_CE_loss(self, h_dec, h_dec_kd, lm_pad):
        def CXE(target, predicted):
            return -(target * predicted.log_softmax(-1))
        
        lm_weight = self.kd_ILM_teacher_weight
        
        mask = lm_pad != self.ignore_id
        mask = mask.to(lm_pad.device)
        
        dec_logits = self.joint_network.forward_ILM(h_dec).squeeze(2)
        dec_logits = dec_logits[:,:-1,1:] # remove last token and blank
        h_dec_kd = h_dec_kd[:,:,1:] # remove blank
        assert h_dec_kd.shape == dec_logits.shape
        #loss_fc = torch.nn.CrossEntropyLoss(reduction='none')
        #loss = loss_fc(de)
        h_dec_kd *= lm_weight # lm weight
        loss = CXE(torch.softmax(h_dec_kd.reshape(-1, h_dec_kd.size(-1)), dim=-1), dec_logits.reshape(-1, dec_logits.size(-1)))
        loss = loss * mask.view(-1,1)
        
        count = mask.sum()
        
        return loss.sum()/count
    
    def lm_gt_loss(self, h_dec, ys_pad):
        mask = ys_pad != self.ignore_id
        mask = mask.to(ys_pad.device)
        dec_logits = self.joint_network.forward_ILM(h_dec).squeeze(2)
        dec_logits = dec_logits[:,:-1,:].contiguous()
        assert dec_logits.shape[1] == ys_pad.shape[1]
        
        ys_pad[ys_pad == self.ignore_id] = 0
        ys_pad = ys_pad[...,:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(dec_logits.view(-1, dec_logits.size(-1)), ys_pad.view(-1))
        
        logp = loss * mask.view(-1)
        logp = logp.sum()
        count = mask.sum()
        
        return logp / count
    
    def dec_feature_loss(self, pred_pad, lm_kd_pad, ys_pad):
        start = torch.ones(ys_pad.shape[0], 1).int().to(ys_pad.device)
        ys_pad = torch.cat([start, ys_pad], dim=1)
        
        mask = ys_pad != self.ignore_id
        mask = mask.to(pred_pad.device)
        
        def _l1_loss(pred, target):
            loss_fn = torch.nn.L1Loss(reduction='none')
            loss = loss_fn(pred,target)
            return loss
        
        def _l2_loss(pred, target):
            loss_fn = torch.nn.MSELoss(reduction='none')
            loss = loss_fn(pred,target)
            return loss
        def _cosine_sim(pred, target):
            loss_fn = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
            loss = loss_fn(pred, target)
            loss = loss.unsqueeze(-1)
            return loss
        
        loss_fn = _l1_loss
        loss = loss_fn(pred_pad, lm_kd_pad)
        loss = (loss * mask.unsqueeze(-1)).view(-1)
        count = mask.sum()
        
        #return loss.sum() / (count*pred_pad.shape[-1])
        return loss.sum() / (count)
    
    def forward(self, xs_pad, ilens, ys_pad, ys_kd_pad=None):
        """E2E forward.
        Args:
            xs_pad (torch.Tensor): batch of padded source sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        Returns:
            loss (torch.Tensor): transducer loss value
        """
        # 1. encoder
        xs_pad = xs_pad[:, : max(ilens)]

        if "custom" in self.etype:
            src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)

            _hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        else:
            _hs_pad, hs_mask, _ = self.enc(xs_pad, ilens)

        if self.use_aux_task:
            hs_pad, aux_hs_pad = _hs_pad[0], _hs_pad[1]
        else:
            hs_pad, aux_hs_pad = _hs_pad, None

        # 1.5. transducer preparation related
        ys_in_pad, ys_out_pad, target, pred_len, target_len = prepare_loss_inputs(
            ys_pad, hs_mask, ignore_id=self.ignore_id
        )

        # 2. decoder
        if "custom" in self.dtype:
            ys_mask = target_mask(ys_in_pad, self.blank_id)
            pred_pad, _ = self.decoder(ys_in_pad, ys_mask, hs_pad)
        else:
            pred_pad = self.dec(hs_pad, ys_in_pad)


        z = self.joint_network(hs_pad.unsqueeze(2), pred_pad.unsqueeze(1))

        # 3. loss computation
        loss_trans = self.criterion(z, target, pred_len, target_len)
        if self.trans_loss_reduction == "none": # using none reduction
            loss_trans = torch.mean(loss_trans)

        if self.use_aux_task and aux_hs_pad is not None:
            loss_aux_trans, loss_aux_symm_kl = self.auxiliary_task(
                aux_hs_pad, pred_pad, z, target, pred_len, target_len
            )
        else:
            loss_aux_trans, loss_aux_symm_kl = 0.0, 0.0

        if self.use_aux_ctc:
            if "custom" in self.etype:
                hs_mask = torch.IntTensor(
                    [h.size(1) for h in hs_mask],
                ).to(hs_mask.device)

            loss_ctc = self.aux_ctc_weight * self.aux_ctc(hs_pad, hs_mask, ys_pad)
        else:
            loss_ctc = 0.0

        if self.use_aux_cross_entropy:
            loss_lm = self.aux_cross_entropy_weight * self.aux_cross_entropy(
                self.aux_decoder_output(pred_pad), ys_out_pad
            )
        elif self.use_ILM_gt_loss:
            loss_lm = self.lm_gt_loss(pred_pad.unsqueeze(2), ys_pad) * self.ILM_gt_loss_factor
        else:
            loss_lm = 0.0

        loss = (
            loss_trans
            + self.transducer_weight * (loss_aux_trans + loss_aux_symm_kl)
            + loss_ctc
            + loss_lm
        )

        self.loss = loss
        loss_data = float(loss)

        # 4. compute cer/wer
        if self.training or self.error_calculator is None:
            cer, wer = None, None
        else:
            cer, wer = self.error_calculator(hs_pad, ys_pad)

        if not math.isnan(loss_data):
            self.reporter.report(
                loss_data,
                float(loss_trans),
                float(loss_ctc),
                float(loss_lm),
                float(loss_aux_trans),
                float(loss_aux_symm_kl),
                cer,
                wer,
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)

        return self.loss

    def forward_kd(self, xs_pad, ilens, ys_pad, ys_kd_pad, lm_kd_pad=None):
        """E2E forward.
        Args:
            xs_pad (torch.Tensor): batch of padded source sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        Returns:
            loss (torch.Tensor): transducer loss value
        """
        # 1. encoder
        xs_pad = xs_pad[:, : max(ilens)]
        def cal_enc_T(ilens):
            return [(((idim - 1) // 2 - 1) // 2) for idim in ilens]
        if "custom" in self.etype:
            src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)

            _hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        else:
            _hs_pad, hs_mask, _ = self.enc(xs_pad, ilens)

        if self.use_aux_task:
            hs_pad, aux_hs_pad = _hs_pad[0], _hs_pad[1]
        else:
            hs_pad, aux_hs_pad = _hs_pad, None

        # 1.5. transducer preparation related
        ys_in_pad, ys_out_pad, target, pred_len, target_len = prepare_loss_inputs(
            ys_pad, hs_mask,ignore_id=self.ignore_id
        )

        # 2. decoder
        if "custom" in self.dtype:
            ys_mask = target_mask(ys_in_pad, self.blank_id)
            pred_pad, _ = self.decoder(ys_in_pad, ys_mask, hs_pad)
        else:
            pred_pad = self.dec(hs_pad, ys_in_pad)
        
        z = self.joint_network(hs_pad.unsqueeze(2), pred_pad.unsqueeze(1))

        # 3. loss computation
        loss_trans = self.criterion(z, target, pred_len, target_len)

        if self.use_aux_task and aux_hs_pad is not None:
            loss_aux_trans, loss_aux_symm_kl = self.auxiliary_task(
                aux_hs_pad, pred_pad, z, target, pred_len, target_len
            )
        else:
            loss_aux_trans, loss_aux_symm_kl = 0.0, 0.0

        if self.use_aux_ctc:
            if "custom" in self.etype:
                hs_mask = torch.IntTensor(
                    [h.size(1) for h in hs_mask],
                ).to(hs_mask.device)

            loss_ctc = self.aux_ctc_weight * self.aux_ctc(hs_pad, hs_mask, ys_pad)
        else:
            loss_ctc = 0.0

        if self.use_aux_cross_entropy:
            loss_lm = self.aux_cross_entropy_weight * self.aux_cross_entropy(
                self.aux_decoder_output(pred_pad), ys_out_pad
            )
        else:
            loss_lm = 0.0
        
        if self.do_kd:           
            if self.use_dec_feature_loss:
                #logging.warning('Using decoder feature loss')
                dec_kd_pad = ys_kd_pad if lm_kd_pad is None else lm_kd_pad
                #logging.warning(dec_kd_pad.shape)
                loss_lm += self.dec_feature_loss(pred_pad, dec_kd_pad, ys_pad) * self.dec_feature_loss_factor
            elif (lm_kd_pad is not None and self.kd_ILM_loss_factor > 0):
                logging.warning('Using ILM CXE loss')
                lm_kd_pad = ys_kd_pad if lm_kd_pad is None else lm_kd_pad
                loss_lm += self.kd_ILM_loss_factor*self.kd_ILM_CE_loss(pred_pad.unsqueeze(2), lm_kd_pad, ys_pad)
            if self.kd_mtl_factor > 0:
                loss_kd = self.kd_mtl_factor * self.kd_loss(z, pred_len, target_len, cal_enc_T(ilens), ys_pad, ys_kd_pad)
            else:
                loss_kd = 0.0
        else:
            loss_kd = 0.0


        loss = (
            loss_trans
            + self.transducer_weight * (loss_aux_trans + loss_aux_symm_kl)
            + loss_ctc
            + loss_lm
            + loss_kd
        )

        self.loss = loss
        loss_data = float(loss)

        # 4. compute cer/wer
        if self.training or self.error_calculator is None:
            cer, wer = None, None
        else:
            cer, wer = self.error_calculator(hs_pad, ys_pad)

        if not math.isnan(loss_data):
            self.reporter.report_kd(
                loss_data,
                float(loss_trans),
                float(loss_ctc),
                float(loss_lm),
                float(loss_aux_trans),
                float(loss_aux_symm_kl),
                float(loss_kd),
                cer,
                wer,
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)

        return self.loss

    def encode_custom(self, x):
        """Encode acoustic features.
        Args:
            x (ndarray): input acoustic feature (T, D)
        Returns:
            x (torch.Tensor): encoded features (T, D_enc)
        """
        p = next(self.parameters())
        x = torch.as_tensor(x,device=p.device).unsqueeze(0)
        enc_output, _ = self.encoder(x, None)

        return enc_output.squeeze(0)

    def encode_rnn(self, x):
        """Encode acoustic features.
        Args:
            x (ndarray): input acoustic feature (T, D)
        Returns:
            x (torch.Tensor): encoded features (T, D_enc)
        """
        p = next(self.parameters())
        if self.etype == 'wav2vec':
            self.enc.encoders.mask_channel_prob = 0
            self.enc.encoders.mask_prob = 0

        if len(x.shape) > 1: # filter bank features
            ilens = [x.shape[0]]
            x = x[:: self.subsample[0], :]
        else: # waveform features
            ilens = [x.shape[0]]
            x = x[:: self.subsample[0]]

        h = torch.as_tensor(x, device=p.device, dtype=p.dtype)
        hs = h.contiguous().unsqueeze(0)

        hs, _, _ = self.enc(hs, ilens)

        return hs.squeeze(0)

    def recognize(self, x, beam_search):
        """Recognize input features.
        Args:
            x (ndarray): input acoustic feature (T, D)
            beam_search (class): beam search class
        Returns:
            nbest_hyps (list): n-best decoding results
        """
        self.eval()

        if "custom" in self.etype:
            h = self.encode_custom(x)
        else:
            h = self.encode_rnn(x)

        nbest_hyps = beam_search(h)

        return [asdict(n) for n in nbest_hyps]

    def get_joint_network_output(self, xs_pad, ys_pad, ilens, calculate_loss=False):
        xs_pad = xs_pad[:, : max(ilens)]
        if "custom" in self.etype:
            src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)

            _hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        else:
            _hs_pad, hs_mask, _ = self.enc(xs_pad, ilens)

        if self.use_aux_task:
            hs_pad, aux_hs_pad = _hs_pad[0], _hs_pad[1]
        else:
            hs_pad, aux_hs_pad = _hs_pad, None

        ys_in_pad, ys_out_pad, target, pred_len, target_len = prepare_loss_inputs(
            ys_pad, hs_mask, ignore_id=self.ignore_id
        )

        if "custom" in self.dtype:
            ys_mask = target_mask(ys_in_pad, self.blank_id)
            pred_pad, _ = self.decoder(ys_in_pad, ys_mask, hs_pad)
        else:
            pred_pad = self.dec(hs_pad, ys_in_pad)
        z = self.joint_network(hs_pad.unsqueeze(2), pred_pad.unsqueeze(1))

        if calculate_loss:
            loss_trans = self.criterion(z, target, pred_len, target_len)
        else:
            loss_trans = 0.0

        return z, loss_trans

    def get_one_best_lm_logits(self, ys_pad, lm):
        prev_token = torch.full((1, ), self.blank, dtype=torch.long, device=ys_pad.device)
        lm_states = None
        lm_score_list = []
        
        lm_states , lm_scores = lm.predict(lm_states, prev_token)
        lm_score_list.append(lm_scores[0])
        
        ys_pad = ys_pad.unsqueeze()
        for token in ys_pad:
            lm_score = lm.predict(lm_states, prev_token)
            lm_score_list.append(lm_score)
            prev_token = token
        
        #lm_score_list = lm_score_list[:-1]
        
        return torch.tensor(lm_score_list)
    
    def viterbi_decoding(self, xs_pad, ilens, ys_pad, lm=None, lm_weight=0.0):
        self.eval()
        if lm is not None:
            use_lm = True
        else:
            use_lm = False
        z, _ = self.get_joint_network_output(xs_pad, ys_pad, ilens)
        if use_lm:
            lm_score = self.get_one_best_lm_logits(ys_pad, lm)*lm_weight
            z = z + lm_score
        
    def collect_soft_label_one_best_lattice(self, xs_pad, ilens, ys_pad, lm=None, lm_weight=0.0):
        self.eval()
        if lm is not None:
            use_lm = True
        else:
            use_lm = False
        z, loss_trans = self.get_joint_network_output(xs_pad, ys_pad, ilens, calculate_loss=True)
        logp = -loss_trans
        
        one_best_path = [0]
        one_best_path_pr = []
        u = 0
        t = 0
        score = 0.0
        lm_state = None
        i = 0
        prev_token = torch.full((1, ), self.blank_id, dtype=torch.long, device=xs_pad.device)
        if use_lm:
            lm_state, lm_scores = lm.predict(lm_state, prev_token)
        while True:
            k = torch.argmax(z[0,t,u,:], dim=-1)
            log_pr = torch.max(z[0,t,u,:].softmax(dim=-1), dim=-1)[0].log()
            score += log_pr
            one_best_path.append(k)
            if use_lm:
                if k == 0: # blank symbol
                    lm_scores[0,0] = 0.0 # set the blank symbol score to zero
                    lm_fused_logit = z[0, t, u, :].log_softmax(0) + lm_weight*lm_scores[0]
                    one_best_path_pr.append(lm_fused_logit)
                    pass # no change of lm_state and lm_score
                else: # update lm score
                    lm_scores[0,0] = min(lm_scores[0]) # set the blank symbol score to the smallest value
                    lm_fused_logit = z[0, t, u, :].log_softmax(0) + lm_weight*lm_scores[0]
                    one_best_path_pr.append(lm_fused_logit)
                    prev_token = torch.full((1, ), k, dtype=torch.long, device=xs_pad.device)
                    lm_state, lm_scores = lm.predict(lm_state, prev_token)
                #if i > 0:
                #normalised = (z[0, t, u, :].log_softmax(0) + lm_weight*lm_scores[0])
                #else:
                #    normalised = z[0, t, u, :].log_softmax(0)
                #    i += 1
                #one_best_path_pr.append(normalised)
            else:
                one_best_path_pr.append(z[0, t, u, :].log_softmax(0))
            if k == 0: # output a blank symbol
                t += 1
            else: # output one symbol
                u += 1
            if t >= z.shape[1]:
                break
            if u >= z.shape[2]:
                break
            if t == z.shape[1] -1 and u == z.shape[2] -1:
                break
        one_best_path = torch.tensor(one_best_path)
        one_best_path_pr = torch.stack(one_best_path_pr)
        yseq = one_best_path[one_best_path > 0]
        return [{'yseq': yseq.cpu().numpy(), 'yseq_with_blank': one_best_path.cpu().numpy(), 'yseq_with_blank_pr': one_best_path_pr.cpu().numpy(), 'score': logp}]

    def collect_soft_label_reduced_lattice(self, xs_pad, ilens, ys_pad):
        self.eval()
        xs_pad = xs_pad[:, : max(ilens)]
        if "custom" in self.etype:
            src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)

            _hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        else:
            _hs_pad, hs_mask, _ = self.enc(xs_pad, ilens)

        if self.use_aux_task:
            hs_pad, aux_hs_pad = _hs_pad[0], _hs_pad[1]
        else:
            hs_pad, aux_hs_pad = _hs_pad, None

        ys_in_pad, ys_out_pad, target, pred_len, target_len = prepare_loss_inputs(
            ys_pad, hs_mask, ignore_id=self.ignore_id
        )

        if "custom" in self.dtype:
            ys_mask = target_mask(ys_in_pad, self.blank_id)
            pred_pad, _ = self.decoder(ys_in_pad, ys_mask, hs_pad)
        else:
            pred_pad = self.dec(hs_pad, ys_in_pad)
        z = self.joint_network(hs_pad.unsqueeze(2), pred_pad.unsqueeze(1))
        pr = z.softmax(dim=-1).cpu().numpy()
        assert z.shape[0] == 1
        reduced_lattice = numpy.zeros((z.shape[0],z.shape[1], z.shape[2]-1, 3))
        for u in range(0,z.shape[2]-1):
            correct_symbol = ys_pad[0,u]
            reduced_lattice[0, :, u, 0] = pr[0, :, u, 0]
            reduced_lattice[0, :, u, 1] = pr[0, :, u, correct_symbol]
            reduced_lattice[0, :, u, 2] = 1 - reduced_lattice[0, :, u, 0] - reduced_lattice[0, :, u, 1]
        return reduced_lattice

    def collect_soft_label_full_lattice(self, xs_pad, ilens, ys_pad):
        self.eval()
        xs_pad = xs_pad[:, : max(ilens)]
        if "custom" in self.etype:
            src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)

            _hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        else:
            _hs_pad, hs_mask, _ = self.enc(xs_pad, ilens)

        if self.use_aux_task:
            hs_pad, aux_hs_pad = _hs_pad[0], _hs_pad[1]
        else:
            hs_pad, aux_hs_pad = _hs_pad, None

        ys_in_pad, ys_out_pad, target, pred_len, target_len = prepare_loss_inputs(
            ys_pad, hs_mask, ignore_id=self.ignore_id
        )

        if "custom" in self.dtype:
            ys_mask = target_mask(ys_in_pad, self.blank_id)
            pred_pad, _ = self.decoder(ys_in_pad, ys_mask, hs_pad)
        else:
            pred_pad = self.dec(hs_pad, ys_in_pad)
        z = self.joint_network(hs_pad.unsqueeze(2), pred_pad.unsqueeze(1))
        pr = z.softmax(dim=-1).cpu().numpy()
        assert z.shape[0] == 1
        return pr

    def collect_decoder_features(self, xs_pad, ilens, ys_pad, lm=None, lm_weight=0.0):
        
        # here assume the batch size is one
        assert xs_pad.shape[0] == 1

        hs_pad = torch.ones(1,1,768).to(ys_pad.device)
        blank = ys_pad.new([0]).view(-1,1)
        ys_pad = torch.cat([blank, ys_pad], -1)
        
        predictor_features = self.dec(hs_pad, ys_pad)
        
        return predictor_features
    
    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.
        Args:
            xs_pad (torch.Tensor): batch of padded input sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor):
                batch of padded character id sequence tensor (B, Lmax)
        Returns:
            ret (ndarray): attention weights with the following shape,
                1) multi-head case => attention weights (B, H, Lmax, Tmax),
                2) other case => attention weights (B, Lmax, Tmax).
        """
        self.eval()

        if "custom" not in self.etype and "custom" not in self.dtype:
            return []
        else:
            with torch.no_grad():
                self.forward(xs_pad, ilens, ys_pad)

            ret = dict()
            for name, m in self.named_modules():
                if isinstance(m, MultiHeadedAttention) or isinstance(
                    m, RelPositionMultiHeadedAttention
                ):
                    ret[name] = m.attn.cpu().numpy()

        self.train()

        return ret
    
    def forward_ILM(self, x, t): 
        # forward ILM by feeding input and target t

        hs_mask = torch.ones(x.size(0),1,320).bool()

        ys_in_pad, ys_out_pad, target, pred_len, target_len = prepare_loss_inputs(
            x, hs_mask,ignore_id=self.ignore_id
        )
        
        hs_pad = torch.randn(x.size(0), 320, 144).to(x.device)
        
        pred_pad = self.dec(hs_pad, x)
        
        mask = x != self.ignore_id
        mask = mask.to(x.device)
        
        x[x == self.ignore_id] = 0
        ys_pad = x[...,:].contiguous()
            
        dec_logits = self.joint_network.forward_ILM(pred_pad)
        shift_logits = dec_logits[..., :, :].contiguous() 
        shift_labels = t[..., :].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        logp = loss * mask.view(-1)
        logp = logp.sum()
        count = mask.sum()
        
        return logp / count, logp, count
    
    def collect_encoder_features(self, xs_pad, ilens, ys_pad):
        if self.etype == 'wav2vec':
            self.enc.encoders.mask_channel_prob = 0
            self.enc.encoders.mask_prob = 0
            
        xs_pad = xs_pad[:, : max(ilens)]
        
        if "custom" in self.etype:
            src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)

            _hs_pad, _ = self.encoder(xs_pad, src_mask)
        else:
            _hs_pad, _, _ = self.enc(xs_pad, ilens)
        
        hs_pad, _  = _hs_pad, None
        
        return hs_pad
    
    def forward_encoder_KD(self, xs_pad, ilens, ys_pad, enc_kd_pad=None):
        """Encoder forwarding for KD

        Args:
            xs_pad (torch.Tensor): batch of padded source sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
            enc_kd_pad (torch.Tensor): batch of padded encoder output as target in KD training

        Returns:
            _type_: _description_
        """
        enc_kd_pad = ys_pad if enc_kd_pad is None else enc_kd_pad
        
        bs = xs_pad.size(0)
        xs_pad = xs_pad[:, : max(ilens)]

        if "custom" in self.etype:
            src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)

            _hs_pad, _ = self.encoder(xs_pad, src_mask)
        else:
            _hs_pad, _, _ = self.enc(xs_pad, ilens)
        
        hs_pad, _  = _hs_pad, None
        
        criterion = torch.nn.L1Loss(reduction='none')
        min_T = min(enc_kd_pad.shape[1] , hs_pad.shape[1])
        enc_kd_pad = enc_kd_pad[:, :min_T, :]
        hs_pad = hs_pad[:, :min_T, :]
        
        mask = enc_kd_pad != self.ignore_id
        
        loss_kd = criterion(hs_pad, enc_kd_pad) * mask
        if self.training:
            loss_kd = loss_kd.sum()/(bs*hs_pad.shape[-1])
            loss_kd *= self.kd_mtl_factor
        else:
            loss_kd = loss_kd.sum()/(mask.sum()*hs_pad.shape[-1])
        
        loss_lm = 0.0
        loss_trans = 0.0
        loss_ctc = 0.0
        loss_aux_trans, loss_aux_symm_kl = 0.0, 0.0
        
        loss = loss_trans + loss_kd
        
        self.loss = loss
        loss_data = float(loss)
        
        cer, wer = 0.0, 0.0
        
        if not math.isnan(loss_data):
            self.reporter.report_kd(
                loss_data,
                float(loss_trans),
                float(loss_ctc),
                float(loss_lm),
                float(loss_aux_trans),
                float(loss_aux_symm_kl),
                float(loss_kd),
                cer,
                wer,
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)

        return self.loss
    
    def forward_decoder_KD(self, xs_pad, ilens, ys_pad, dec_kd_pad=None):
        # forward decoder for KD    
        bs = ys_pad.size(0)
        dec_kd_pad = xs_pad if dec_kd_pad is None else dec_kd_pad
        
        hs_pad = torch.ones(bs,1,768).to(ys_pad.device)
        ys = [y[y != self.ignore_id] for y in ys_pad]
        blank = ys[0].new([0])

        ys_in_pad = pad_list([torch.cat([blank, y], dim=0) for y in ys], 0)
        
        predictor_features = self.dec(hs_pad, ys_in_pad)
        
        mask = ys_pad != self.ignore_id
        mask = torch.cat([torch.ones(bs, 1).bool().to(mask.device), mask], dim=1)
        criterion = torch.nn.L1Loss(reduction='none')
        loss_kd = criterion(predictor_features, dec_kd_pad) * mask.unsqueeze(-1)
        
        if self.training:
            loss_kd = loss_kd.sum()/(bs*predictor_features.shape[-1])
            loss_kd *= self.kd_mtl_factor
        else:
            loss_kd = loss_kd.sum()/(mask.sum()*predictor_features.shape[-1])
        
        loss_lm = 0.0
        loss_trans = 0.0
        loss_ctc = 0.0
        loss_aux_trans, loss_aux_symm_kl = 0.0, 0.0
        
        loss = loss_trans + loss_kd
        
        self.loss = loss
        loss_data = float(loss)
        
        cer, wer = 0.0, 0.0
        
        if not math.isnan(loss_data):
            self.reporter.report_kd(
                loss_data,
                float(loss_trans),
                float(loss_ctc),
                float(loss_lm),
                float(loss_aux_trans),
                float(loss_aux_symm_kl),
                float(loss_kd),
                cer,
                wer,
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
            
        return self.loss
        
    def forward_nbest_KD(self, xs_pad: torch.Tensor, ilens: torch.Tensor, n_best_list: List, prob_list=None):
        """E2E n-best KD forward

        Args:
            xs_pad (torch.Tensor): batch of padded input 
            ilens (torch.Tensor): batch of lengths of input sequences
            n_best_list (list): a list of tuples, each tuple consists of (transcription, kd_aligment), total length = n
        """
        bs = xs_pad.shape[0]
        xs_pad = xs_pad[:, : max(ilens)]
        def cal_enc_T(ilens):
            return [(((idim - 1) // 2 - 1) // 2) for idim in ilens]
        if "custom" in self.etype:
            src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)

            _hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        else:
            _hs_pad, hs_mask, _ = self.enc(xs_pad, ilens)

        if self.use_aux_task:
            hs_pad, aux_hs_pad = _hs_pad[0], _hs_pad[1]
        else:
            hs_pad, aux_hs_pad = _hs_pad, None
            
        n_best_num = len(n_best_list)
        
        loss_aux_trans, loss_aux_symm_kl = 0.0, 0.0
        loss_ctc, loss_lm = 0.0, 0.0
        loss_kd = 0.0
        loss_trans = 0.0
        loss_trans_list = []
        loss_kd_list = []
        hyp_probs = torch.from_numpy(numpy.array(prob_list)).float().to(xs_pad.device).softmax(1)
        
        for i, (ys_pad, kd_pad) in enumerate(n_best_list):
            # prepare transducer loss
            ys_in_pad, ys_out_pad, target, pred_len, target_len = prepare_loss_inputs(ys_pad, hs_mask,ignore_id=self.ignore_id)
            
            if "custom" in self.dtype:
                ys_mask = target_mask(ys_in_pad, self.blank_id)
                pred_pad, _ = self.decoder(ys_in_pad, ys_mask, hs_pad)
            else:
                pred_pad = self.dec(hs_pad, ys_in_pad)
            
            # lattice generation
            z = self.joint_network(hs_pad.unsqueeze(2), pred_pad.unsqueeze(1))
            
            # loss computation
            loss_trans += torch.sum(self.criterion(z, target, pred_len, target_len) * hyp_probs[:, i])
            
            if self.kd_mtl_factor > 0:
                temp_loss = self.kd_loss(z, pred_len, target_len, cal_enc_T(ilens), ys_pad, kd_pad, reduction='none')
                for j in range(len(temp_loss)):
                    loss_kd += temp_loss[j] * hyp_probs[j, i]
                #loss_kd_list += self.kd_loss(z, pred_len, target_len, cal_enc_T(ilens), ys_pad, kd_pad, reduction='none')
        
        loss_trans = loss_trans / bs
        loss_kd = loss_kd * self.kd_mtl_factor / bs
                
        loss = (
            loss_trans
            + self.transducer_weight * (loss_aux_trans + loss_aux_symm_kl)
            + loss_ctc
            + loss_lm
            + loss_kd
        )

        self.loss = loss
        loss_data = float(loss)

        # compute cer/wer
        if self.training or self.error_calculator is None:
            cer, wer = None, None
        else:
            cer, wer = self.error_calculator(hs_pad, ys_pad)
        
        if not math.isnan(loss_data):
            self.reporter.report_kd(
                    loss_data,
                    float(loss_trans),
                    float(loss_ctc),
                    float(loss_lm),
                    float(loss_aux_trans),
                    float(loss_aux_symm_kl),
                    float(loss_kd),
                    cer,
                    wer,
                )   
        else:
            logging.warning("loss (=%f) is not correct", loss_data)    

        return self.loss

    def forward_nbest_KD_efficient(self, xs_pad: torch.Tensor, ilens: torch.Tensor, n_best_list: List, prob_list=None):
        """E2E n-best KD forward

        Args:
            xs_pad (torch.Tensor): batch of padded input 
            ilens (torch.Tensor): batch of lengths of input sequences
            n_best_list (list): a list of tuples, each tuple consists of (transcription, kd_aligment), total length = n
        """
        bs = xs_pad.shape[0]
        xs_pad = xs_pad[:, : max(ilens)]
        def cal_enc_T(ilens):
            return [(((idim - 1) // 2 - 1) // 2) for idim in ilens]
        if "custom" in self.etype:
            src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)

            _hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        else:
            _hs_pad, hs_mask, _ = self.enc(xs_pad, ilens)

        if self.use_aux_task:
            hs_pad, aux_hs_pad = _hs_pad[0], _hs_pad[1]
        else:
            hs_pad, aux_hs_pad = _hs_pad, None
            
        n_best_num = len(n_best_list)
        
        loss_aux_trans, loss_aux_symm_kl = 0.0, 0.0
        loss_ctc, loss_lm = 0.0, 0.0
        loss_kd = 0.0
        loss_trans = 0.0
        loss_trans_list = []
        loss_kd_list = []
        hyp_probs = torch.from_numpy(numpy.array(prob_list)).float().to(xs_pad.device).softmax(1)
        
        for i, (ys_pad, kd_pad) in enumerate(n_best_list):
            # prepare transducer loss
            ys_in_pad, ys_out_pad, target, pred_len, target_len = prepare_loss_inputs(ys_pad, hs_mask,ignore_id=self.ignore_id)
            
            if "custom" in self.dtype:
                ys_mask = target_mask(ys_in_pad, self.blank_id)
                pred_pad, _ = self.decoder(ys_in_pad, ys_mask, hs_pad)
            else:
                pred_pad = self.dec(hs_pad, ys_in_pad)
            
            # lattice generation
            z = self.joint_network(hs_pad.unsqueeze(2), pred_pad.unsqueeze(1))
            
            # loss computation
            loss_trans += torch.sum(self.criterion(z, target, pred_len, target_len) * hyp_probs[:, i])
            
            if self.kd_mtl_factor > 0:
                temp_loss = self.kd_loss(z, pred_len, target_len, cal_enc_T(ilens), ys_pad, kd_pad, reduction='none')
                for j in range(len(temp_loss)):
                    loss_kd += temp_loss[j] * hyp_probs[j, i]
                #loss_kd_list += self.kd_loss(z, pred_len, target_len, cal_enc_T(ilens), ys_pad, kd_pad, reduction='none')
        
        loss_trans = loss_trans / bs
        loss_kd = loss_kd * self.kd_mtl_factor / bs
                
        loss = (
            loss_trans
            + self.transducer_weight * (loss_aux_trans + loss_aux_symm_kl)
            + loss_ctc
            + loss_lm
            + loss_kd
        )

        self.loss = loss
        loss_data = float(loss)

        # compute cer/wer
        if self.training or self.error_calculator is None:
            cer, wer = None, None
        else:
            cer, wer = self.error_calculator(hs_pad, ys_pad)
        
        if not math.isnan(loss_data):
            self.reporter.report_kd(
                    loss_data,
                    float(loss_trans),
                    float(loss_ctc),
                    float(loss_lm),
                    float(loss_aux_trans),
                    float(loss_aux_symm_kl),
                    float(loss_kd),
                    cer,
                    wer,
                )   
        else:
            logging.warning("loss (=%f) is not correct", loss_data)    

        return self.loss