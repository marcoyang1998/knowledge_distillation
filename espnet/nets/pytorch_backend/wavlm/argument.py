"""WavLM encoder common arguments"""

from distutils.util import strtobool
import logging

def add_arguments_wavlm_common(group):
    group.add_argument(
        "--wavlm-model-dir",
        type=str,
        help="The "
    )
    group.add_argument(
        "--wavlm-output-dim",
        type=int,
        help="wavlm output size"
    )
    group.add_argument(
        "--wavlm-freeze-finetune-updates",
        type=int,
        default=0,
        help="How many freezing updates"
    )
    group.add_argument(
        "--wavlm-mask-channel-prob",
        type=float,
        default=0.5,
        help="wavlm mask channel prob"
    )
    group.add_argument(
        "--wavlm-mask-channel-length",
        type=int,
        default=64,
        help="Mask channel length"
    )
    group.add_argument(
        "--wavlm-mask-prob",
        type=float,
        default=0.65,
        help="wavlm mask channel prob"
    )
    group.add_argument(
        "--wavlm-subsample",
        type=strtobool,
        default=True,
        help="Add a subsample module after wavlm encoder"
    )
    group.add_argument(
        "--wavlm-subsample-mode",
        type=str,
        default='concat-tanh',
        help="Subsampling mode"
    )

    return group
