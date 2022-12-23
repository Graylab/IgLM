from typing import List

import numpy as np
from tqdm import tqdm

from iglm.model.tokens import *


def iglm_to_infilled(token_seq: List, tokenizer):
    """
    Convert IgLM inputs to the infilled tokenized sequence without any special tokens.
    """
    token_seq = np.array(token_seq)
    sep_token_idx = np.nonzero(token_seq == tokenizer.sep_token_id)[0].item()
    cls_token_idx = np.nonzero(token_seq == tokenizer.cls_token_id)[0].min()
    mask_token_idx = np.nonzero(token_seq == tokenizer.mask_token_id)[0].item()

    infilled_seq = np.concatenate([
        token_seq[:mask_token_idx], token_seq[sep_token_idx + 1:cls_token_idx],
        token_seq[mask_token_idx + 1:sep_token_idx]
    ],
                                  axis=0)

    # Remove any conditioning tokens, which have token IDs greater than or equal to 25
    infilled_seq = infilled_seq[infilled_seq < 25]

    return infilled_seq


def mask_span(seq: List, start: int, end: int, append_span: bool = False):
    masked_seq = seq[:start] + [MASK_TOKEN] + seq[end:] + [SEP_TOKEN]
    if append_span:
        masked_seq += seq[start:end]

    return masked_seq


def validate_generated_seq(input_ids, tokenizer):
    """
    Validate that generated input ids are well formed ([MASK] before [SEP] and [CLS] and that there's one of each).
    """
    mask_idx = np.where(input_ids == tokenizer.mask_token_id)[0]
    sep_idx = np.where(input_ids == tokenizer.sep_token_id)[0]
    cls_idx = np.where(input_ids == tokenizer.cls_token_id)[0]

    if len(mask_idx) != 1 or len(sep_idx) != 1 or len(cls_idx) != 1:
        return False

    mask_idx = mask_idx.squeeze()
    sep_idx = sep_idx.squeeze()
    cls_idx = cls_idx.squeeze()

    return (mask_idx < sep_idx) and (sep_idx < cls_idx)