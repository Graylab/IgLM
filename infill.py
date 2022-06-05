import argparse
import os
from datetime import datetime

import torch
import transformers
from Bio import SeqIO

from iglm.infill_utils import BAD_WORD_IDS, mask_span, iglm_generate
from util import set_seeds


def main():
    """
    Generate antibody sequences given chain and species tokens.
    """
    now = str(datetime.now().strftime('%y-%m-%d %H:%M:%S'))

    parser = argparse.ArgumentParser(description='Re-design antibody sequence from start position to end position')
    parser.add_argument('fasta_file', type=str, help='Antibody fasta file')
    parser.add_argument('record_id', type=str, help='Record ID in fasta corresponding to the sequence to design')
    parser.add_argument('start', type=int, help='Start index (0-indexed)')
    parser.add_argument('end', type=int, help='End index (0-indexed, exclusive)')
    parser.add_argument('--chkpt_dir', type=str, default='trained_models/IgLM', help='Model checkpoint directory')
    parser.add_argument('--chain_token', type=str, default='[HEAVY]', help='Chain to design (i.e. "heavy" or "light")')
    parser.add_argument('--species_token', type=str, default='[HUMAN]', help='Species')
    parser.add_argument('--num_seqs', type=int, default=1000, help='Number of sequences to generate')
    parser.add_argument('--temperature', type=float, default=1, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=1, help='p for top-p sampling')
    parser.add_argument('--output_dir', type=str, default=f'output_dir/infill_{now}')
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

    # Load seq from fasta
    target_record = list(filter(lambda x: x.id == args.record_id, SeqIO.parse(open(args.fasta_file), 'fasta')))
    assert len(target_record) == 1, f"Expected 1 record in fasta with id {args.record_id}, " \
                                    f"but got {len(target_record)} records"
    wt_seq = list(target_record[0].seq)

    # Get starting tokens
    chain_token = args.chain_token
    species_token = args.species_token

    # Starting tokens
    masked_seq = mask_span(wt_seq, args.start, args.end)  # mask using provided indices
    start_tokens = [chain_token, species_token] + masked_seq
    start_tokens = torch.Tensor([tokenizer.convert_tokens_to_ids(start_tokens)]).int().to(device)
    print(f"Starting tokens: {tokenizer.decode(start_tokens.squeeze())}")

    assert (start_tokens != tokenizer.unk_token_id).all(), "Unrecognized token supplied in starting tokens"

    # Generate seqs
    print("Generating sequences to infill [MASK] token...")
    generated_seqs = iglm_generate(model, start_tokens, tokenizer, num_to_generate=args.num_seqs,
                                   top_p=args.top_p,
                                   temperature=args.temperature,
                                   bad_word_ids=BAD_WORD_IDS)

    # Write seqs to file
    out_fasta_file = f'{args.output_dir}/generated_seqs.fasta'

    with open(out_fasta_file, 'w') as fasta:
        for i, seq in enumerate(generated_seqs):
            print(f'>seq_{i}', file=fasta)
            print(seq, file=fasta)


if __name__ == '__main__':
    main()
