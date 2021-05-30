"""Wav2vec2.0 common arguments"""

from distutils.util import strtobool

def add_arguments_w2v2_common(group):
    """Add Wav2vec2 common arguments."""
    group.add_argument(
        "--w2v2-model-dir",
        type=str,
        help="path to the w2v2 model"
    )
    group.add_argument(
        "--w2v2-normalise-before",
        type=strtobool,
        default=False,
        help="apply layernorm before feeding waveform"
    )
    group.add_argument(
        "--w2v2-freeze-finetune-updates",
        type=int,
        default=1000,
        help="freeze until x updates"
    )
    group.add_argument(
        "--w2v2-output-dim",
        type=int,
        default=1024,
        help="output dimesion"
    )
    group.add_argument(
        "--w2v2-is-finetuned",
        type=strtobool,
        default=False,
        help="whether the model is fine-tuned"
    )