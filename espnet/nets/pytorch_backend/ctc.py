from distutils.version import LooseVersion
import logging

import numpy as np
import six
import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import to_device


class CTC(torch.nn.Module):
    """CTC module

    :param int odim: dimension of outputs
    :param int eprojs: number of encoder projection units
    :param float dropout_rate: dropout rate (0.0 ~ 1.0)
    :param str ctc_type: builtin or warpctc
    :param bool reduce: reduce the CTC loss into a scalar
    """

    def __init__(self, odim, eprojs, dropout_rate, ctc_type="warpctc", reduce=True, ignore_id=-1, do_kd=False, kd_factor=1.0, kd_temp=1.0):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.loss = None
        self.ctc_odim=odim
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.probs = None  # for visualization
        self.do_knowledge_distillation = do_kd

        # In case of Pytorch >= 1.7.0, CTC will be always builtin
        self.ctc_type = (
            ctc_type
            if LooseVersion(torch.__version__) < LooseVersion("1.7.0")
            else "builtin"
        )

        if self.do_knowledge_distillation:
            if self.ctc_type != "builtin":
                raise NotImplementedError("Only support knowledge ditillation with builtin ctc")
            self.kd_factor = kd_factor
            self.kd_temp=kd_temp


        # ctc_type = buitin not support Pytorch=1.0.1
        if self.ctc_type == "builtin" and (
            LooseVersion(torch.__version__) < LooseVersion("1.1.0")
        ):
            self.ctc_type = "cudnnctc"

        if ctc_type != self.ctc_type:
            logging.warning(f"CTC was set to {self.ctc_type} due to PyTorch version.")

        if self.ctc_type == "builtin":
            reduction_type = "sum" if reduce else "none"
            self.ctc_loss = torch.nn.CTCLoss(
                reduction=reduction_type, zero_infinity=True
            )
        elif self.ctc_type == "cudnnctc":
            reduction_type = "sum" if reduce else "none"
            self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)
        elif self.ctc_type == "warpctc":
            import warpctc_pytorch as warp_ctc

            self.ctc_loss = warp_ctc.CTCLoss(size_average=True, reduce=reduce)
        elif self.ctc_type == "gtnctc":
            from espnet.nets.pytorch_backend.gtn_ctc import GTNCTCLossFunction

            self.ctc_loss = GTNCTCLossFunction.apply
        else:
            raise ValueError(
                'ctc_type must be "builtin" or "warpctc": {}'.format(self.ctc_type)
            )

        self.ignore_id = ignore_id
        self.reduce = reduce

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen):
        if self.ctc_type in ["builtin", "cudnnctc"]:
            th_pred = th_pred.log_softmax(2)
            # Use the deterministic CuDNN implementation of CTC loss to avoid
            #  [issue#17798](https://github.com/pytorch/pytorch/issues/17798)
            with torch.backends.cudnn.flags(deterministic=True):
                loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
            # Batch-size average
            loss = loss / th_pred.size(1)
            return loss
        elif self.ctc_type == "warpctc":
            return self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
        elif self.ctc_type == "gtnctc":
            targets = [t.tolist() for t in th_target]
            log_probs = torch.nn.functional.log_softmax(th_pred, dim=2)
            return self.ctc_loss(log_probs, targets, 0, "none")
        else:
            raise NotImplementedError

    def kd_loss_fn(self, logits, soft_logits, hlens):
        # logits (B, adim,odim)
        def CXE(target, predicted):
            return -(target * predicted.log_softmax(-1)).sum()

        bs = logits.shape[0]
        odim = logits.shape[-1]
        loss = 0
        for b in range(bs):
            soft_logit = soft_logits[b].view(-1, odim)
            assert abs(soft_logit.shape[0] - hlens[b]) <= 1, "Softlabel: {}, logit: {}".format(soft_logit.shape[0], hlens[b])
            #assert soft_logit.shape[0] >= hlens[b], "Softlabel: {}, logit: {}".format(soft_logit.shape[0], hlens[b])
            l = min(hlens[b], soft_logit.shape[0])
            loss += CXE((soft_logit[:l, :] / self.kd_temp).softmax(-1), logits[b, :l, :] / self.kd_temp)
        return loss / bs


    def forward(self, hs_pad, hlens, ys_pad):
        """CTC forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad:
            batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        """
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys

        # zero padding for hs
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        if self.ctc_type != "gtnctc":
            ys_hat = ys_hat.transpose(0, 1)

        if self.ctc_type == "builtin":
            hlens = hlens.long()
            olens = to_device(ys_hat, torch.LongTensor([len(s) for s in ys]))
            ys_pad = torch.cat(ys)  # without this the code breaks for asr_mix
            self.loss = self.loss_fn(ys_hat, ys_pad, hlens, olens)
        else:
            self.loss = None
            hlens = torch.from_numpy(np.fromiter(hlens, dtype=np.int32))
            olens = torch.from_numpy(
                np.fromiter((x.size(0) for x in ys), dtype=np.int32)
            )
            # zero padding for ys
            ys_true = torch.cat(ys).cpu().int()  # batch x olen
            # get ctc loss
            # expected shape of seqLength x batchSize x alphabet_size
            dtype = ys_hat.dtype
            if self.ctc_type == "warpctc" or dtype == torch.float16:
                # warpctc only supports float32
                # torch.ctc does not support float16 (#1751)
                ys_hat = ys_hat.to(dtype=torch.float32)
            if self.ctc_type == "cudnnctc":
                # use GPU when using the cuDNN implementation
                ys_true = to_device(hs_pad, ys_true)
            if self.ctc_type == "gtnctc":
                # keep as list for gtn
                ys_true = ys
            self.loss = to_device(
                hs_pad, self.loss_fn(ys_hat, ys_true, hlens, olens)
            ).to(dtype=dtype)

        # get length info
        logging.info(
            self.__class__.__name__
            + " input lengths:  "
            + "".join(str(hlens).split("\n"))
        )
        logging.info(
            self.__class__.__name__
            + " output lengths: "
            + "".join(str(olens).split("\n"))
        )

        if self.reduce:
            # NOTE: sum() is needed to keep consistency
            # since warpctc return as tensor w/ shape (1,)
            # but builtin return as tensor w/o shape (scalar).
            self.loss = self.loss.sum()
            logging.info("ctc loss:" + str(float(self.loss)))

        return self.loss

    def forward_kd(self, hs_pad, hlens, ys_pad, ys_kd_pad):
        ys = [y[y != self.ignore_id] for y in ys_pad]
        ys_kd = [y[y != self.ignore_id] for y in ys_kd_pad]

        # zero padding for hs
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        if self.ctc_type != "gtnctc":
            ys_hat = ys_hat.transpose(0, 1)
        if self.ctc_type == "builtin":
            hlens = hlens.long()
            olens = to_device(ys_hat, torch.LongTensor([len(s) for s in ys]))

            # original ctc loss
            if self.kd_factor != 1.0:
                if torch.sum(olens != 1) ==0:
                    self.loss_ctc = 0
                else:
                    ys_hat_valid = ys_hat[:, olens != 1]
                    hlens_valid = hlens[olens != 1]
                    olens_valid = olens[olens != 1]
                    ys_valid = [k for k in ys if k.shape[-1] != 1]
                    ys_pad_valid = torch.cat(ys_valid)  # without this the code breaks for asr_mix
                    self.loss_ctc = self.loss_fn(ys_hat_valid, ys_pad_valid, hlens_valid, olens_valid)
            else:
                self.loss_ctc = 0
            # distillation loss
            if self.kd_factor != 0:
                self.loss_kd = self.kd_loss_fn(ys_hat.transpose(0, 1), ys_kd, hlens)
            else:
                self.loss_kd = 0

            # cast type of kd_factor
            kd_factor = self.kd_factor
            self.loss = kd_factor * self.loss_kd + (1 - kd_factor) * self.loss_ctc

        else:
            raise NotImplementedError('Types other than builtin are not implemented')

        logging.info(
            self.__class__.__name__
            + " input lengths:  "
            + "".join(str(hlens).split("\n"))
        )
        logging.info(
            self.__class__.__name__
            + " output lengths: "
            + "".join(str(olens).split("\n"))
        )

        if self.reduce:
            # NOTE: sum() is needed to keep consistency
            # since warpctc return as tensor w/ shape (1,)
            # but builtin return as tensor w/o shape (scalar).
            self.loss = self.loss.sum()
            logging.info("kd loss:{}, ctc loss:{}, joint loss:{}".format(self.loss_kd, self.loss_ctc, self.loss))

        return tuple([self.loss_kd, self.loss_ctc, kd_factor])

    def softmax(self, hs_pad):
        """softmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: log softmax applied 3d tensor (B, Tmax, odim)
        :rtype: torch.Tensor
        """
        self.probs = F.softmax(self.ctc_lo(hs_pad), dim=2)
        return self.probs

    def log_softmax(self, hs_pad):
        """log_softmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: log softmax applied 3d tensor (B, Tmax, odim)
        :rtype: torch.Tensor
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad):
        """argmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: argmax applied 2d tensor (B, Tmax)
        :rtype: torch.Tensor
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)

    def forced_align(self, h, y, blank_id=0):
        """forced alignment.

        :param torch.Tensor h: hidden state sequence, 2d tensor (T, D)
        :param torch.Tensor y: id sequence tensor 1d tensor (L)
        :param int y: blank symbol index
        :return: best alignment results
        :rtype: list
        """

        def interpolate_blank(label, blank_id=0):
            """Insert blank token between every two label token."""
            label = np.expand_dims(label, 1)
            blanks = np.zeros((label.shape[0], 1), dtype=np.int64) + blank_id
            label = np.concatenate([blanks, label], axis=1)
            label = label.reshape(-1)
            label = np.append(label, label[0])
            return label

        lpz = self.log_softmax(h)
        lpz = lpz.squeeze(0)

        y_int = interpolate_blank(y, blank_id)

        logdelta = np.zeros((lpz.size(0), len(y_int))) - 100000000000.0  # log of zero
        state_path = (
            np.zeros((lpz.size(0), len(y_int)), dtype=np.int16) - 1
        )  # state path

        logdelta[0, 0] = lpz[0][y_int[0]]
        logdelta[0, 1] = lpz[0][y_int[1]]

        for t in six.moves.range(1, lpz.size(0)):
            for s in six.moves.range(len(y_int)):
                if y_int[s] == blank_id or s < 2 or y_int[s] == y_int[s - 2]:
                    candidates = np.array([logdelta[t - 1, s], logdelta[t - 1, s - 1]])
                    prev_state = [s, s - 1]
                else:
                    candidates = np.array(
                        [
                            logdelta[t - 1, s],
                            logdelta[t - 1, s - 1],
                            logdelta[t - 1, s - 2],
                        ]
                    )
                    prev_state = [s, s - 1, s - 2]
                logdelta[t, s] = np.max(candidates) + lpz[t][y_int[s]]
                state_path[t, s] = prev_state[np.argmax(candidates)]

        state_seq = -1 * np.ones((lpz.size(0), 1), dtype=np.int16)

        candidates = np.array(
            [logdelta[-1, len(y_int) - 1], logdelta[-1, len(y_int) - 2]]
        )
        prev_state = [len(y_int) - 1, len(y_int) - 2]
        state_seq[-1] = prev_state[np.argmax(candidates)]
        for t in six.moves.range(lpz.size(0) - 2, -1, -1):
            state_seq[t] = state_path[t + 1, state_seq[t + 1, 0]]

        output_state_seq = []
        for t in six.moves.range(0, lpz.size(0)):
            output_state_seq.append(y_int[state_seq[t, 0]])

        return output_state_seq

class CTC_kd_mtl(torch.nn.Module):
    def __init__(self, odim, eprojs, dropout_rate,
                 ctc_type="warpctc",
                 reduce=True,
                 kd_factor=1.0):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.loss = None
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.probs = None  # for visualization

        # In case of Pytorch >= 1.7.0, CTC will be always builtin
        self.ctc_type = (
            ctc_type
            if LooseVersion(torch.__version__) < LooseVersion("1.7.0")
            else "builtin"
        )

        if self.ctc_type != "builtin":
            raise NotImplementedError("Only support knowledge ditillation with builtin ctc")

        self.kd_factor = kd_factor
        assert kd_factor <= 1.0 and kd_factor >= 0, "kd factor is invalid, value must be within [0,1], get {}".format(kd_factor)

        if self.ctc_type == "builtin" and (
            LooseVersion(torch.__version__) < LooseVersion("1.1.0")
        ):
            self.ctc_type = "cudnnctc"

        if ctc_type != self.ctc_type:
            logging.warning(f"CTC was set to {self.ctc_type} due to PyTorch version.")

        if self.ctc_type == "builtin":
            reduction_type = "sum" if reduce else "none"
            self.ctc_loss = torch.nn.CTCLoss(
                reduction=reduction_type, zero_infinity=True
            )
        elif self.ctc_type == "cudnnctc":
            reduction_type = "sum" if reduce else "none"
            self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)
        elif self.ctc_type == "warpctc":
            import warpctc_pytorch as warp_ctc

            self.ctc_loss = warp_ctc.CTCLoss(size_average=True, reduce=reduce)
        elif self.ctc_type == "gtnctc":
            from espnet.nets.pytorch_backend.gtn_ctc import GTNCTCLossFunction

            self.ctc_loss = GTNCTCLossFunction.apply
        else:
            raise ValueError(
                'ctc_type must be "builtin" or "warpctc": {}'.format(self.ctc_type)
            )
        self.ignore_id = -1
        self.reduce = reduce

    def ctc_loss_fn(self, th_pred, th_target, th_ilen, th_olen):
        if self.ctc_type in ["builtin", "cudnnctc"]:
            th_pred = th_pred.log_softmax(2)
            # Use the deterministic CuDNN implementation of CTC loss to avoid
            #  [issue#17798](https://github.com/pytorch/pytorch/issues/17798)
            with torch.backends.cudnn.flags(deterministic=True):
                loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
            # Batch-size average
            loss = loss / th_pred.size(1)
            return loss
        elif self.ctc_type == "warpctc":
            return self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
        elif self.ctc_type == "gtnctc":
            targets = [t.tolist() for t in th_target]
            log_probs = torch.nn.functional.log_softmax(th_pred, dim=2)
            return self.ctc_loss(log_probs, targets, 0, "none")
        else:
            raise NotImplementedError

    def kd_loss_fn(self, logits, soft_logits, hlens):
        # logits (B, adim,odim)
        def CXE(target, predicted):
            return -(target * predicted.log_softmax(-1)).sum()

        bs = logits.shape[0]
        odim = logits.shape[-1]
        loss = 0
        for b in range(bs):
            soft_logit = soft_logits[b].view(-1, odim)
            assert soft_logit.shape[0] >= hlens[b]
            loss += CXE((soft_logit[:hlens[b], :] / self.kd_temp).softmax(-1), logits[b, :hlens[b], :] / self.kd_temp)
        return loss / bs

    def forward(self, hs_pad, hlens, ys_pad, ys_kd_pad):
        ys = [y[y != self.ignore_id] for y in ys_pad]
        ys_kd = [y[y != self.ignore_id] for y in ys_kd_pad]

        # zero padding for hs
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        if self.ctc_type != "gtnctc":
            ys_hat = ys_hat.transpose(0, 1)
        if self.ctc_type == "builtin":
            hlens = hlens.long()
            olens = to_device(ys_hat, torch.LongTensor([len(s) for s in ys]))
            # original ctc loss
            if self.kd_factor != 1.0:
                ys_pad = torch.cat(ys)  # without this the code breaks for asr_mix
                self.loss_ctc = self.ctc_loss_fn(ys_hat, ys_pad, hlens, olens)
            # distillation loss
            if self.kd_factor != 0:
                self.loss_kd = self.kd_loss_fn(ys_hat.transpose(0, 1), ys_kd, hlens)
            else:
                self.loss_kd = 0
            # cast type of kd_factor
            kd_factor = torch.tensor(self.kd_factor).float().to(self.loss_ctc.device)
            self.loss = kd_factor*self.loss_kd + (1-kd_factor)*self.loss_ctc
        else:
            raise NotImplementedError('Types other than builtin are not implemented')

        logging.info(
            self.__class__.__name__
            + " input lengths:  "
            + "".join(str(hlens).split("\n"))
        )
        logging.info(
            self.__class__.__name__
            + " output lengths: "
            + "".join(str(olens).split("\n"))
        )

        if self.reduce:
            # NOTE: sum() is needed to keep consistency
            # since warpctc return as tensor w/ shape (1,)
            # but builtin return as tensor w/o shape (scalar).
            self.loss = self.loss.sum()
            logging.info("kd loss:{}, ctc loss:{}, joint loss:{}".format(self.loss_kd, self.loss_ctc, self.loss))

        return tuple([self.loss_kd, self.loss_ctc, kd_factor])

    def softmax(self, hs_pad):
        """softmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: log softmax applied 3d tensor (B, Tmax, odim)
        :rtype: torch.Tensor
        """
        self.probs = F.softmax(self.ctc_lo(hs_pad), dim=2)
        return self.probs

    def log_softmax(self, hs_pad):
        """log_softmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: log softmax applied 3d tensor (B, Tmax, odim)
        :rtype: torch.Tensor
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad):
        """argmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: argmax applied 2d tensor (B, Tmax)
        :rtype: torch.Tensor
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)

    def forced_align(self, h, y, blank_id=0):
        """forced alignment.

        :param torch.Tensor h: hidden state sequence, 2d tensor (T, D)
        :param torch.Tensor y: id sequence tensor 1d tensor (L)
        :param int y: blank symbol index
        :return: best alignment results
        :rtype: list
        """

        def interpolate_blank(label, blank_id=0):
            """Insert blank token between every two label token."""
            label = np.expand_dims(label, 1)
            blanks = np.zeros((label.shape[0], 1), dtype=np.int64) + blank_id
            label = np.concatenate([blanks, label], axis=1)
            label = label.reshape(-1)
            label = np.append(label, label[0])
            return label

        lpz = self.log_softmax(h)
        lpz = lpz.squeeze(0)

        y_int = interpolate_blank(y, blank_id)

        logdelta = np.zeros((lpz.size(0), len(y_int))) - 100000000000.0  # log of zero
        state_path = (
            np.zeros((lpz.size(0), len(y_int)), dtype=np.int16) - 1
        )  # state path

        logdelta[0, 0] = lpz[0][y_int[0]]
        logdelta[0, 1] = lpz[0][y_int[1]]

        for t in six.moves.range(1, lpz.size(0)):
            for s in six.moves.range(len(y_int)):
                if y_int[s] == blank_id or s < 2 or y_int[s] == y_int[s - 2]:
                    candidates = np.array([logdelta[t - 1, s], logdelta[t - 1, s - 1]])
                    prev_state = [s, s - 1]
                else:
                    candidates = np.array(
                        [
                            logdelta[t - 1, s],
                            logdelta[t - 1, s - 1],
                            logdelta[t - 1, s - 2],
                        ]
                    )
                    prev_state = [s, s - 1, s - 2]
                logdelta[t, s] = np.max(candidates) + lpz[t][y_int[s]]
                state_path[t, s] = prev_state[np.argmax(candidates)]

        state_seq = -1 * np.ones((lpz.size(0), 1), dtype=np.int16)

        candidates = np.array(
            [logdelta[-1, len(y_int) - 1], logdelta[-1, len(y_int) - 2]]
        )
        prev_state = [len(y_int) - 1, len(y_int) - 2]
        state_seq[-1] = prev_state[np.argmax(candidates)]
        for t in six.moves.range(lpz.size(0) - 2, -1, -1):
            state_seq[t] = state_path[t + 1, state_seq[t + 1, 0]]

        output_state_seq = []
        for t in six.moves.range(0, lpz.size(0)):
            output_state_seq.append(y_int[state_seq[t, 0]])

        return output_state_seq

def ctc_for(args, odim, reduce=True):
    """Returns the CTC module for the given args and output dimension

    :param Namespace args: the program args
    :param int odim : The output dimension
    :param bool reduce : return the CTC loss in a scalar
    :return: the corresponding CTC module
    """
    num_encs = getattr(args, "num_encs", 1)  # use getattr to keep compatibility
    if num_encs == 1:
        # compatible with single encoder asr mode
        do_kd = getattr(args, "do_knowledge_distillation", False)
        kd_mtl_factor = getattr(args, "kd_mtl_factor", 1.0)
        return CTC(
            odim, args.eprojs, args.dropout_rate, ctc_type=args.ctc_type, reduce=reduce, do_kd=do_kd, kd_factor=kd_mtl_factor
        )
    elif num_encs >= 1:
        ctcs_list = torch.nn.ModuleList()
        if args.share_ctc:
            # use dropout_rate of the first encoder
            ctc = CTC(
                odim,
                args.eprojs,
                args.dropout_rate[0],
                ctc_type=args.ctc_type,
                reduce=reduce,
            )
            ctcs_list.append(ctc)
        else:
            for idx in range(num_encs):
                ctc = CTC(
                    odim,
                    args.eprojs,
                    args.dropout_rate[idx],
                    ctc_type=args.ctc_type,
                    reduce=reduce,
                )
                ctcs_list.append(ctc)
        return ctcs_list
    else:
        raise ValueError(
            "Number of encoders needs to be more than one. {}".format(num_encs)
        )
