import argparse
import os
from datetime import datetime

import torch
import transformers

from iglm import IgLM
from iglm.utils.seed import set_seeds


def main():
    """
    Generate full antibody sequences given chain and species tokens.
    """
    now = str(datetime.now().strftime('%y-%m-%d %H:%M:%S'))

    parser = argparse.ArgumentParser(
        description='Generate antibody sequences from scratch')
    parser.add_argument(
        '--model',
        type=str,
        default="IgLM",
        help='Model to use (IgLM, IgLM-S',
    )
    parser.add_argument(
        '--prompt_sequence',
        type=str,
        help='Prompt sequence to start generation from',
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
        default=f'output_dir/full_generation_{now}',
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

    # Get starting tokens
    chain_token = args.chain_token
    species_token = args.species_token
    prompt_sequence = args.prompt_sequence

    # Get sampling parameters
    temperature = args.temperature
    top_p = args.top_p
    num_to_generate = args.num_seqs

    # Generate designed seqs
    print(f"Generating sequences with {species_token} {chain_token} tokens...")
    generated_seqs = iglm.generate(
        chain_token,
        species_token,
        prompt_sequence=prompt_sequence,
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
