from typing import List

import numpy as np
from tqdm import tqdm

BAD_WORD_IDS = [[25], [26], [27], [28], [29], [30], [31], [32]]
MASK_TOKEN = '[MASK]'
SEP_TOKEN = '[SEP]'
ANS_TOKEN = '[ANS]'

SPECIES_TO_TOKEN = {
    "Camel": "[CAMEL]",
    "human": "[HUMAN]",
    "HIS-mouse": "[MOUSE]",
    "mouse_Balb/c": "[MOUSE]",
    "mouse_BALB/c": "[MOUSE]",
    "mouse_C57BL/6": "[MOUSE]",
    "mouse_C57BL/6J": "[MOUSE]",
    "mouse_Ighe/e": "[MOUSE]",
    "mouse_Ighg/g": "[MOUSE]",
    "mouse_Igh/wt": "[MOUSE]",
    "mouse_outbred": "[MOUSE]",
    "mouse_outbred/C57BL/6": "[MOUSE]",
    "mouse_RAG2-GFP/129Sve": "[MOUSE]",
    "mouse_Swiss-Webster": "[MOUSE]",
    "rabbit": "[RABBIT]",
    "rat": "[RAT]",
    "rat_SD": "[RAT]",
    "rhesus": "[RHESUS]",
}

CHAIN_TO_TOKEN = {
    "Heavy": "[HEAVY]",
    "Light": "[LIGHT]"
}


def iglm_to_infilled(token_seq: List, tokenizer):
    """
    Convert IgLM inputs to the infilled tokenized sequence without any special tokens.
    """
    token_seq = np.array(token_seq)
    sep_token_idx = np.nonzero(token_seq == tokenizer.sep_token_id)[0].item()
    cls_token_idx = np.nonzero(token_seq == tokenizer.cls_token_id)[0].min()
    mask_token_idx = np.nonzero(token_seq == tokenizer.mask_token_id)[0].item()

    infilled_seq = np.concatenate([token_seq[:mask_token_idx], token_seq[sep_token_idx + 1:cls_token_idx],
                   token_seq[mask_token_idx + 1:sep_token_idx]], axis=0)

    # Remove any conditioning tokens, which have token IDs greater than or equal to 25
    infilled_seq = infilled_seq[infilled_seq < 25]

    return infilled_seq


def mask_span(seq: List, start: int, end: int):
    return seq[:start] + [MASK_TOKEN] + seq[end:] + [SEP_TOKEN]


def iglm_generate(model, starting_tokens, tokenizer, num_to_generate, top_p, temperature, bad_word_ids):
    decoded_seqs = set()  # Set to remove duplicates
    pbar = tqdm(total=num_to_generate)
    while len(decoded_seqs) < num_to_generate:
        seq = model.generate(starting_tokens,
                             max_length=150,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.cls_token_id,
                             forced_eos_token_id=tokenizer.cls_token_id,
                             bad_words_ids=bad_word_ids,
                             do_sample=True,
                             top_p=top_p,
                             temperature=temperature
                             ).detach().cpu().numpy()

        seq = seq[0]  # Squeeze out batch dimension
        if validate_generated_seq(seq, tokenizer):
            decoded_seq = ''.join(tokenizer.decode(iglm_to_infilled(seq, tokenizer))).replace(' ', '')
            if decoded_seq not in decoded_seqs:
                decoded_seqs.add(decoded_seq)
                pbar.update(1)

    pbar.close()
    return list(decoded_seqs)


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