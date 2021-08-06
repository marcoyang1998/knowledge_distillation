"""Transducer speech recognition model (pytorch)."""

from argparse import Namespace
from collections import Counter
from dataclasses import asdict
import logging
import math
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
from espnet.nets.pytorch_backend.transducer.auxiliary_task import AuxiliaryTask
from espnet.nets.pytorch_backend.transducer.custom_decoder import CustomDecoder
from espnet.nets.pytorch_backend.transducer.custom_encoder import CustomEncoder
from espnet.nets.pytorch_backend.transducer.error_calculator import ErrorCalculator
from espnet.nets.pytorch_backend.transducer.initializer import initializer
from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork
from espnet.nets.pytorch_backend.transducer.loss import TransLoss
from espnet.nets.pytorch_backend.transducer.rnn_decoder import DecoderRNNT
from espnet.nets.pytorch_backend.transducer.rnn_encoder import encoder_for
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

    def __init__(self, idim, odim, args, ignore_id=-1, blank_id=0, training=True):
        """Construct an E2E object for transducer model."""
        torch.nn.Module.__init__(self)

        args = fill_missing_args(args, self.add_arguments)

        self.is_rnnt = True
        self.transducer_weight = args.transducer_weight

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
                streaming=args.streaming
            )
            encoder_out = self.encoder.enc_out

            self.most_dom_list = args.enc_block_arch[:]
        else:
            self.subsample = get_subsample(args, mode="asr", arch="rnn-t")

            self.enc = encoder_for(
                args,
                idim,
                self.subsample,
                aux_task_layer_list=aux_task_layer_list,
            )
            encoder_out = args.eprojs

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
                ignore_id=ignore_id
            )
            decoder_out = args.dunits

        self.joint_network = JointNetwork(
            odim, encoder_out, decoder_out, args.joint_dim, args.joint_activation_type
        )

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
        self.ignore_id = ignore_id
        print("Ignore_id: {}".format(self.ignore_id))

        self.space = args.sym_space
        self.blank = args.sym_blank

        self.odim = odim
        self.do_kd = args.do_knowledge_distillation

        self.reporter = Reporter()

        self.error_calculator = None

        self.default_parameters(args)

        if training:
            self.criterion = TransLoss(args.trans_type, self.blank_id)

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
                    ignore_id=ignore_id
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
                    odim, ignore_id, args.aux_cross_entropy_smoothing
                )

            if self.do_kd:
                self.kd_mtl_factor = args.kd_mtl_factor
                self.kd_temperature = args.kd_temperature
                self.kd_mode = args.transducer_kd_mode
                if self.kd_mode == "one_best_path":
                    self.kd_loss = self.kd_one_best_loss
                elif self.kd_mode == "reduced_lattice":
                    self.kd_loss = self.kd_reduced_lattice_loss
                elif self.kd_mode == "shifted_one_best_path":
                    self.kd_loss = self.kd_shifted_one_best_loss
                    self.shift_step = args.shift_step
                else:
                    raise NotImplementedError("Not implemented {}".format(self.kd_mode))
                print("Distillation mode: {}, kd factor: {}".format(self.kd_mode, self.kd_mtl_factor))
        self.loss = None
        self.rnnlm = None
        if args.streaming:
            print("This is a streaming transducer!")

    def default_parameters(self, args):
        """Initialize/reset parameters for transducer.

        Args:
            args (Namespace): argument Namespace containing options

        """
        initializer(self, args)

    def kd_one_best_loss(self, z, pred_len, target_len, enc_T, ys_pad, ys_pad_kd):
        # 0: t_list, 1: u_list， 2： y_seq_with_blank
        def CXE(target, predicted):
            return -(target * predicted.log_softmax(-1)).sum()

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

        ys_kd = torch.cat(ys_kd, dim=0)
        logits = torch.cat(logits, dim=0)
        kd_loss = CXE(torch.softmax(ys_kd / self.kd_temperature, dim=-1), logits / self.kd_temperature)

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

    def forward(self, xs_pad, ilens, ys_pad):
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

    def forward_kd(self, xs_pad, ilens, ys_pad, ys_kd_pad):
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

        if len(x.shape) > 1:
            ilens = [x.shape[0]]
            x = x[:: self.subsample[0], :]
        else:
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

    def collect_soft_label_one_best_lattice(self, xs_pad, ilens, ys_pad):
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
        one_best_path = [0]
        one_best_path_pr = []
        u = 0
        t = 0
        score = 0.0
        while True:
            k = torch.argmax(z[0,t,u,:], dim=-1)
            log_pr = torch.max(z[0,t,u,:].softmax(dim=-1), dim=-1)[0].log()
            score += log_pr
            one_best_path.append(k)
            one_best_path_pr.append(z[0,t,u,:])
            if k == 0:
                t += 1
            else:
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
        return [{'yseq': yseq.cpu().numpy(), 'yseq_with_blank': one_best_path.cpu().numpy(), 'yseq_with_blank_pr': one_best_path_pr.cpu().numpy(), 'score': score.cpu().numpy()}]

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
