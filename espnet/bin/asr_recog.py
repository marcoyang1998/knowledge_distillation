#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""End-to-end speech recognition model decoding script."""

import configargparse
import logging
import os
import random
import sys

import numpy as np

from espnet.utils.cli_utils import strtobool

# NOTE: you need this func to generate our sphinx doc


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transcribe text from speech using "
        "a speech recognition model on one CPU or GPU",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="Config file path")
    parser.add(
        "--config2",
        is_config_file=True,
        help="Second config file path that overwrites the settings in `--config`",
    )
    parser.add(
        "--config3",
        is_config_file=True,
        help="Third config file path that overwrites the settings "
        "in `--config` and `--config2`",
    )

    parser.add_argument("--ngpu", type=int, default=0, help="Number of GPUs")
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32", "float64"),
        default="float32",
        help="Float precision (only available in --api v2)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="chainer",
        choices=["chainer", "pytorch"],
        help="Backend library",
    )
    parser.add_argument("--debugmode", type=int, default=1, help="Debugmode")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--verbose", "-V", type=int, default=1, help="Verbose option")
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1,
        help="Batch size for beam search (0: means no batch processing)",
    )
    parser.add_argument(
        "--preprocess-conf",
        type=str,
        default=None,
        help="The configuration file for the pre-processing",
    )
    parser.add_argument(
        "--api",
        default="v1",
        choices=["v1", "v2"],
        help="Beam search APIs "
        "v1: Default API. It only supports the ASRInterface.recognize method "
        "and DefaultRNNLM. "
        "v2: Experimental API. It supports any models that implements ScorerInterface.",
    )
    # task related
    parser.add_argument(
        "--recog-json", type=str, help="Filename of recognition data (json)"
    )
    parser.add_argument(
        "--result-label",
        type=str,
        required=True,
        help="Filename of result label data (json)",
    )
    # model (parameter) related
    parser.add_argument(
        "--model", type=str, required=True, help="Model file parameters to read"
    )
    parser.add_argument(
        "--model-conf", type=str, default=None, help="Model config file"
    )
    parser.add_argument(
        "--num-spkrs",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of speakers in the speech",
    )
    parser.add_argument(
        "--num-encs", default=1, type=int, help="Number of encoders in the model."
    )
    # search related
    parser.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size")
    parser.add_argument("--penalty", type=float, default=0.0, help="Incertion penalty")
    parser.add_argument(
        "--maxlenratio",
        type=float,
        default=0.0,
        help="""Input length ratio to obtain max output length.
                        If maxlenratio=0.0 (default), it uses a end-detect function
                        to automatically find maximum hypothesis lengths""",
    )
    parser.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain min output length",
    )
    parser.add_argument(
        "--ctc-weight", type=float, default=0.0, help="CTC weight in joint decoding"
    )
    parser.add_argument(
        "--weights-ctc-dec",
        type=float,
        action="append",
        help="ctc weight assigned to each encoder during decoding."
        "[in multi-encoder mode only]",
    )
    parser.add_argument(
        "--ctc-window-margin",
        type=int,
        default=0,
        help="""Use CTC window with margin parameter to accelerate
                        CTC/attention decoding especially on GPU. Smaller magin
                        makes decoding faster, but may increase search errors.
                        If margin=0 (default), this function is disabled""",
    )
    # transducer related
    parser.add_argument(
        "--search-type",
        type=str,
        default="default",
        choices=["default", "nsc", "tsd", "alsd", "ILME", "non_duplicated"],
        help="""Type of beam search implementation to use during inference.
        Can be either: default beam search, n-step constrained beam search ("nsc"),
        time-synchronous decoding ("tsd") or alignment-length synchronous decoding
        ("alsd").
        Additional associated parameters: "nstep" + "prefix-alpha" (for nsc),
        "max-sym-exp" (for tsd) and "u-max" (for alsd)""",
    )
    parser.add_argument(
        "--nstep",
        type=int,
        default=1,
        help="Number of expansion steps allowed in NSC beam search.",
    )
    parser.add_argument(
        "--prefix-alpha",
        type=int,
        default=2,
        help="Length prefix difference allowed in NSC beam search.",
    )
    parser.add_argument(
        "--max-sym-exp",
        type=int,
        default=2,
        help="Number of symbol expansions allowed in TSD decoding.",
    )
    parser.add_argument(
        "--u-max",
        type=int,
        default=400,
        help="Length prefix difference allowed in ALSD beam search.",
    )
    parser.add_argument(
        "--score-norm",
        type=strtobool,
        nargs="?",
        default=True,
        help="Normalize transducer scores by length",
    )
    # rnnlm related
    parser.add_argument(
        "--rnnlm", type=str, default=None, help="RNNLM model file to read"
    )
    parser.add_argument(
        "--rnnlm-conf", type=str, default=None, help="RNNLM model config file to read"
    )
    parser.add_argument(
        "--word-rnnlm", type=str, default=None, help="Word RNNLM model file to read"
    )
    parser.add_argument(
        "--word-rnnlm-conf",
        type=str,
        default=None,
        help="Word RNNLM model config file to read",
    )
    parser.add_argument("--word-dict", type=str, default=None, help="Word list to read")
    parser.add_argument("--lm-weight", type=float, default=0.1, help="RNNLM weight")
    # ngram related
    parser.add_argument(
        "--ngram-model", type=str, default=None, help="ngram model file to read"
    )
    parser.add_argument("--ngram-weight", type=float, default=0.1, help="ngram weight")
    parser.add_argument(
        "--ngram-scorer",
        type=str,
        default="part",
        choices=("full", "part"),
        help="""if the ngram is set as a part scorer, similar with CTC scorer,
                ngram scorer only scores topK hypethesis.
                if the ngram is set as full scorer, ngram scorer scores all hypthesis
                the decoding speed of part scorer is musch faster than full one""",
    )
    # streaming related
    parser.add_argument(
        "--streaming-mode",
        type=str,
        default=None,
        choices=["window", "segment"],
        help="""Use streaming recognizer for inference.
                        `--batchsize` must be set to 0 to enable this mode""",
    )
    parser.add_argument("--streaming-window", type=int, default=10, help="Window size")
    parser.add_argument(
        "--streaming-min-blank-dur",
        type=int,
        default=10,
        help="Minimum blank duration threshold",
    )
    parser.add_argument(
        "--streaming-onset-margin", type=int, default=1, help="Onset margin"
    )
    parser.add_argument(
        "--streaming-offset-margin", type=int, default=1, help="Offset margin"
    )
    # non-autoregressive related
    # Mask CTC related. See https://arxiv.org/abs/2005.08700 for the detail.
    parser.add_argument(
        "--maskctc-n-iterations",
        type=int,
        default=10,
        help="Number of decoding iterations."
        "For Mask CTC, set 0 to predict 1 mask/iter.",
    )
    parser.add_argument(
        "--maskctc-probability-threshold",
        type=float,
        default=0.999,
        help="Threshold probability for CTC output",
    )
    parser.add_argument(
        "--collect-soft-label",
        type=strtobool,
        default=False,
        help="If true, only collect ctc-labels and don't perform actual recognition"
    )
    parser.add_argument(
        "--output-kd-dir",
        type=str,
        help="Where to store the collected kd label"
    )
    parser.add_argument(
        "--collect-rnnt-kd-data",
        type=strtobool,
        default=False,
        help="If true, collect distillation data for rnnt model while performing decoding"
    )
    parser.add_argument(
        "--rnnt-kd-data-collection-mode",
        type=str,
        choices=["beam_search", "reduced_lattice", "one_best_lattice", "full_lattice", "decoder_logits", "decoder_features", "encoder_features"],
        default= "beam_search",
        help="Beam search: using beam search to find the best sequence, Reduced lattice: paper's implementation, one best lattice: using ground truth to find best path"

    )
    parser.add_argument(
        "--keep-gt-transcription",
        type=strtobool,
        default=False,
        help="If true, pseudo transcription will be written. Only set this to true when using unlabelled data"
    )
    parser.add_argument(
        "--collect-rnnlm-blank",
        type=strtobool,
        default=False,
        help="collect rnnlm blank logit"
    )
    parser.add_argument(
        "--rnnlm-blank-logit-dir",
        type=str,
        help="where to store the blank logit"
    )
    parser.add_argument(
        "--collect-rnnlm-logit",
        type=strtobool,
        default=False,
        help="collect rnnlm logit"
    )
    parser.add_argument(
        "--rnnlm-logit-dir",
        type=str,
        help="where to store the blank logit"
    )
    parser.add_argument(
        "--lm-fusion-kd",
        type=strtobool,
        default=False,
        help="If true, store shallow fused prob with LM integration"
    )
    parser.add_argument(
        "--kd-json-label",
        type=str,
        help="Name of the kd json file"
    )
    parser.add_argument(
        "--internal-lm-weight",
        type=float,
        default=0.0,
        help="If 0, ILME is deactivated. Otherwise the ILM score is subtracted from the original score"
    )
    parser.add_argument(
        "--calculate-ILM-ppl",
        type=strtobool,
        default=False,
        help="calculate the ppl of transducer ILM"
    )
    parser.add_argument(
        "--ILM-valid-label",
        type=str,
        help="path to valid.txt file"
    )
    parser.add_argument(
        "--collect-lm-feature",
        type=strtobool,
        default=False,
        help="collect feature vectors before linear output layer"
    )
    parser.add_argument(
        "--allow-duplications",
        type=strtobool,
        default=True,
        help="allow duplications in beam search"
    )
    return parser


def main(args):
    """Run the main decoding function."""
    parser = get_parser()
    args = parser.parse_args(args)

    if args.ngpu == 0 and args.dtype == "float16":
        raise ValueError(f"--dtype {args.dtype} does not support the CPU backend.")

    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose == 2:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

        # TODO(mn5k): support of multiple GPUs
        if args.ngpu > 1:
            logging.error("The program only supports ngpu=1.")
            sys.exit(1)

    # display PYTHONPATH
    logging.info("python path = " + os.environ.get("PYTHONPATH", "(None)"))

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    logging.info("set random seed = %d" % args.seed)

    # validate rnn options
    if args.rnnlm is not None and args.word_rnnlm is not None:
        logging.error(
            "It seems that both --rnnlm and --word-rnnlm are specified. "
            "Please use either option."
        )
        sys.exit(1)

    # recog
    logging.info("backend = " + args.backend)
    if args.num_spkrs == 1:
        if args.backend == "chainer":
            from espnet.asr.chainer_backend.asr import recog

            recog(args)
        elif args.backend == "pytorch":

            if args.num_encs == 1:
                # Experimental API that supports custom LMs
                if args.api == "v2":
                    from espnet.asr.pytorch_backend.recog import recog_v2

                    recog_v2(args)
                elif args.collect_soft_label:
                    from espnet.asr.pytorch_backend.asr import collect_soft_labels
                    logging.info("Do label collection")
                    collect_soft_labels(args)
                elif args.collect_rnnlm_blank:
                    from espnet.asr.pytorch_backend.asr import collect_rnnlm_blank_logit
                    logging.info("Do rnnlm blank logit collection")
                    collect_rnnlm_blank_logit(args)
                elif args.collect_rnnlm_logit:
                    from espnet.asr.pytorch_backend.asr import collect_rnnlm_logit
                    logging.info("Do rnnlm logit collection")
                    collect_rnnlm_logit(args)
                elif args.calculate_ILM_ppl:
                    from espnet.asr.pytorch_backend.asr import calculate_ILM_ppl
                    logging.info("Calculate transducer ILM ppl")
                    calculate_ILM_ppl(args)
                else:
                    from espnet.asr.pytorch_backend.asr import recog

                    if args.dtype != "float32":
                        raise NotImplementedError(
                            f"`--dtype {args.dtype}` is only available with `--api v2`"
                        )
                    recog(args)
            else:
                if args.api == "v2":
                    raise NotImplementedError(
                        f"--num-encs {args.num_encs} > 1 is not supported in --api v2"
                    )
                else:
                    from espnet.asr.pytorch_backend.asr import recog

                    recog(args)
        else:
            raise ValueError("Only chainer and pytorch are supported.")
    elif args.num_spkrs == 2:
        if args.backend == "pytorch":
            from espnet.asr.pytorch_backend.asr_mix import recog

            recog(args)
        else:
            raise ValueError("Only pytorch is supported.")


if __name__ == "__main__":
    main(sys.argv[1:])
