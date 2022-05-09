import argparse
import os
from collections import defaultdict
from datetime import datetime

import datasets
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
from iglm.infill_utils import SPECIES_TO_TOKEN, CHAIN_TO_TOKEN

from util import set_seeds


def main():
    """
    Evalute IgLM on paired human heavy and light chain sequences by computing infilling perplexity as a function of
    span position along each sequence.
    """
    now = str(datetime.now().strftime('%y-%m-%d %H:%M:%S'))

    parser = argparse.ArgumentParser(description='Test model predictions with alanine scans')
    parser.add_argument('chkpt_dir', type=str, help='Model checkpoint directory')
    parser.add_argument('paired_csv_file', type=str, help='Paired sequence csv')
    parser.add_argument('--output_dir', type=str, default=f'output_dir/scan_evaluate_{now}')
    parser.add_argument('--mask_len', type=int, default=10)
    args = parser.parse_args()

    # Reproducibility
    set_seeds()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = transformers.GPT2LMHeadModel.from_pretrained(args.chkpt_dir).to(device)
    model.eval()

    vocab_file = './vocab.txt'
    tokenizer = transformers.BertTokenizerFast(vocab_file=vocab_file, do_lower_case=False)

    # Preprocess dataset
    paired_dataset = pd.read_csv(args.paired_csv_file)
    paired_dataset = paired_dataset[paired_dataset['species'] == 'human']

    # Perform clean and alanine window scanning
    with h5py.File(f'{args.output_dir}/losses.h5', 'w') as h5_out:
        chain_to_loss_per_sample = defaultdict(list)
        chain_to_pos_per_sample = defaultdict(list)

        # For each example, compute loss of all window positions and store in loss_per_sample
        for _, row in tqdm(paired_dataset.iterrows(), total=paired_dataset.shape[0]):
            for chain in ['Heavy', 'Light']:
                batch = build_scan_batch(row, chain, tokenizer, args.mask_len)

                with torch.no_grad():
                    batch['input_ids'], batch['labels'] = batch['input_ids'].to(device), batch['labels'].to(device)
                    outputs = model(batch['input_ids'], labels=batch['labels'])
                    loss_per_sample_in_batch = compute_loss_per_sample(outputs['logits'], batch['labels']).cpu().detach().numpy()
                    pos_per_sample_in_batch = np.arange(loss_per_sample_in_batch.shape[0])  # positions are ordered within each batch since each sample is a position
                    chain_to_loss_per_sample[chain].append(loss_per_sample_in_batch)
                    chain_to_pos_per_sample[chain].append(pos_per_sample_in_batch)

        for chain in ['Heavy', 'Light']:
            loss_per_sample = np.concatenate(chain_to_loss_per_sample[chain])
            pos_per_sample = np.concatenate(chain_to_pos_per_sample[chain])

            loss_at_each_pos = np.zeros(np.max(pos_per_sample) + 1)
            num_samples_at_each_pos = np.zeros(np.max(pos_per_sample) + 1)
            np.add.at(num_samples_at_each_pos, pos_per_sample, 1)
            np.add.at(loss_at_each_pos, pos_per_sample, loss_per_sample)

            avg_loss_per_pos = loss_at_each_pos / num_samples_at_each_pos
            h5_out.create_dataset(f'{chain}_losses', data=avg_loss_per_pos)
            h5_out.create_dataset(f'{chain}_num_samples_per_pos', data=num_samples_at_each_pos)


def build_scan_batch(row, chain, tokenizer, mask_len):
    if chain == 'Heavy':
        seq = list(row['hseq'])
    else:
        seq = list(row['lseq'])

    seqs = []
    chain_token, species_token = CHAIN_TO_TOKEN[chain], SPECIES_TO_TOKEN['human']
    for start in range(0, len(seq) - mask_len):
        end = start + mask_len
        seq_i = tokenizer.convert_tokens_to_ids([chain_token, species_token] + seq[:start] +
                                                [tokenizer.mask_token] + seq[end:] +
                                                [tokenizer.sep_token] + seq[start:end] +
                                                [tokenizer.cls_token])
        seqs.append(seq_i)

    batch = {'input_ids': torch.Tensor(seqs).long()}

    # Mask out amino acids which are not "infilled" for loss computation
    labels = batch["input_ids"].clone()
    labels[:, :-(mask_len + 1)] = -100
    batch["labels"] = labels

    return batch


def compute_loss_per_sample(lm_logits, labels):
    """
    Compute CE loss per sample in batch.
    """
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    losses = losses.view(shift_labels.shape)
    mask = (shift_labels != -100)
    losses = losses * mask
    loss_per_sample = losses.sum(dim=1) / mask.sum(dim=1)
    return loss_per_sample


if __name__ == '__main__':
    main()