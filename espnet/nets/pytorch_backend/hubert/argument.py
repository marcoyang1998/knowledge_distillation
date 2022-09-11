"""Hubert encoder common arguments"""

from distutils.util import strtobool
import logging

def add_arguments_hubert_common(group):
    group.add_argument(
        "--hubert-model-dir",
        type=str,
        help="The hubert model dir"
    )
    group.add_argument(
        "--hubert-output-dim",
        type=int,
        help="Hubert output size"
    )
    group.add_argument(
        "--hubert-freeze-finetune-updates",
        type=int,
        default=0,
        help="How many freezing updates"
    )
    group.add_argument(
        "--hubert-mask-channel-prob",
        type=float,
        default=0.5,
        help="Hubert mask channel prob"
    )
    group.add_argument(
        "--hubert-mask-channel-length",
        type=int,
        default=64,
        help="Mask channel length"
    )
    group.add_argument(
        "--hubert-mask-prob",
        type=float,
        default=0.65,
        help="Hubert mask channel prob"
    )
    group.add_argument(
        "--hubert-apply-mask",
        type=strtobool,
        default=True,
        help="Apply random masking after CNN extractor"
    )
    group.add_argument(
        "--hubert-subsample",
        type=strtobool,
        default=True,
        help="Add a subsample module after hubert encoder"
    )
    group.add_argument(
        "--hubert-subsample-mode",
        type=str,
        default='concat-tanh',
        help="Subsampling mode"
    )

    return group
