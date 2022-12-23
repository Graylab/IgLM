import argparse
import os
from datetime import datetime

import torch
import transformers
from Bio import SeqIO

from iglm import IgLM
from iglm.utils.seed import set_seeds


def main():
    """
    Generate antibody sequences given chain and species tokens.
    """
    now = str(datetime.now().strftime('%y-%m-%d %H:%M:%S'))

    parser = argparse.ArgumentParser(
        description=
        'Re-design antibody sequence from start position to end position', )
    parser.add_argument(
        'fasta_file',
        type=str,
        help='Antibody fasta file',
    )
    parser.add_argument(
        'record_id',
        type=str,
        help='Record ID in fasta corresponding to the sequence to design',
    )
    parser.add_argument(
        'start',
        type=int,
        help='Start index (0-indexed)',
    )
    parser.add_argument(
        'end',
        type=int,
        help='End index (0-indexed, exclusive)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default="IgLM",
        help='Model to use (IgLM, IgLM-S',
    )
    parser.add_argument(
        '--chain_token',
        type=str,
        default='[HEAVY]',
        help='Chain to design (i.e. "heavy" or "light")',
    )
    parser.add_argument(
        '--species_token',
        type=str,
        default='[HUMAN]',
        help='Species',
    )
    parser.add_argument(
        '--num_seqs',
        type=int,
        default=1000,
        help='Number of sequences to generate',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1,
        help='Sampling temperature',
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=1,
        help='p for top-p sampling',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=f'output_dir/infill_{now}',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed',
    )
    args = parser.parse_args()

    # Reproducibility
    set_seeds(args.seed)

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)

    iglm = IgLM(model_name=args.model)

    # Load seq from fasta
    target_record = list(
        filter(lambda x: x.id == args.record_id,
               SeqIO.parse(open(args.fasta_file), 'fasta')))
    assert len(target_record) == 1, f"Expected 1 record in fasta with id {args.record_id}, " \
                                    f"but got {len(target_record)} records"
    parent_seq = list(target_record[0].seq)

    # Get starting tokens
    chain_token = args.chain_token
    species_token = args.species_token

    # Get sampling parameters
    temperature = args.temperature
    top_p = args.top_p
    num_to_generate = args.num_seqs
    infill_range = (args.start, args.end)

    # Generate seqs
    print("Generating sequences to infill [MASK] token...")
    generated_seqs = iglm.infill(
        parent_seq,
        chain_token,
        species_token,
        infill_range=infill_range,
        num_to_generate=num_to_generate,
        top_p=top_p,
        temperature=temperature,
    )

    # Write seqs to file
    out_fasta_file = f'{args.output_dir}/generated_seqs.fasta'

    with open(out_fasta_file, 'w') as fasta:
        for i, seq in enumerate(generated_seqs):
            print(f'>seq_{i}', file=fasta)
            print(seq, file=fasta)


if __name__ == '__main__':
    main()
