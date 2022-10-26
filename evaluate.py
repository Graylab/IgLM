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
        'Calculate log likelihood for an antibody sequence given chain and species tokens'
    )
    # parser.add_argument('fasta_file', type=str, help='Antibody fasta file')
    # parser.add_argument(
    #     'record_id',
    #     type=str,
    #     help='Record ID in fasta corresponding to the sequence to design')
    # parser.add_argument('start', type=int, help='Start index (0-indexed)')
    # parser.add_argument('end',
    #                     type=int,
    #                     help='End index (0-indexed, exclusive)')
    parser.add_argument('--chkpt_dir',
                        type=str,
                        default='trained_models/IgLM',
                        help='Model checkpoint directory')
    parser.add_argument('--chain_token',
                        type=str,
                        default='[HEAVY]',
                        help='Chain to design (i.e. "heavy" or "light")')
    parser.add_argument('--species_token',
                        type=str,
                        default='[HUMAN]',
                        help='Species')
    parser.add_argument('--output_dir',
                        type=str,
                        default=f'output_dir/infill_{now}')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    # Reproducibility
    set_seeds(args.seed)

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)

    iglm = IgLM(chkpt_dir=args.chkpt_dir)

    args.fasta_file = "/Users/jeffreyruffolo/Local/DeepH3_stuff/benchmarks/therapeutic_benchmark/fastas/1bey.fasta"
    args.record_id = "1bey:H"
    args.start = 90
    args.end = 102

    # Load seq from fasta
    target_record = list(
        filter(lambda x: x.id == args.record_id,
               SeqIO.parse(open(args.fasta_file), 'fasta')))
    assert len(target_record) == 1, f"Expected 1 record in fasta with id {args.record_id}, " \
                                    f"but got {len(target_record)} records"
    parent_seq = list(target_record[0].seq)

    # infill_range = (args.start, args.end)
    infill_range = None

    # Get starting tokens
    chain_token = args.chain_token
    species_token = args.species_token

    # Generate seqs
    print("Generating sequences to infill [MASK] token...")
    score = iglm.log_likelihood(
        chain_token,
        species_token,
        parent_seq,
        infill_range=infill_range,
    )

    print(f"Score: {score}")


if __name__ == '__main__':
    main()
