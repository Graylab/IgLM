import argparse
import math

from Bio import SeqIO

from iglm import IgLM
from iglm.utils.general import exists
from iglm.utils.seed import set_seeds


def main():
    """
    Generate antibody sequences given chain and species tokens.
    """

    parser = argparse.ArgumentParser(
        description=
        'Calculate log likelihood for an antibody sequence given chain and species tokens',
    )
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
        '--start',
        type=int,
        default=None,
        help='Start index (0-indexed)',
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='End index (0-indexed, exclusive)',
    )
    parser.add_argument(
        '--chkpt_dir',
        type=str,
        default=None,
        help='Model checkpoint directory',
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
        '--seed',
        type=int,
        default=0,
        help='Random seed',
    )
    args = parser.parse_args()

    # Reproducibility
    set_seeds(args.seed)

    # Load model
    iglm = IgLM(chkpt_dir=args.chkpt_dir)

    # Load seq from fasta
    target_record = list(
        filter(lambda x: x.id == args.record_id,
               SeqIO.parse(open(args.fasta_file), 'fasta')))
    assert len(target_record) == 1, f"Expected 1 record in fasta with id {args.record_id}, " \
                                    f"but got {len(target_record)} records"
    parent_seq = list(target_record[0].seq)

    # Get infill range if specified
    if exists(args.start) and exists(args.end):
        infill_range = (args.start, args.end)
    else:
        infill_range = None

    # Get starting tokens
    chain_token = args.chain_token
    species_token = args.species_token

    # Score sequence
    parent_seq_str = ''.join(parent_seq)
    if exists(infill_range):
        print(
            f"Scoring subsequence: {parent_seq_str[infill_range[0]:infill_range[1]]}\n"
            +
            f"given {parent_seq_str[:infill_range[0]]}[MASK]{parent_seq_str[infill_range[1]:]}"
        )
    else:
        print(f"Scoring sequence: {parent_seq_str}")

    ll = iglm.log_likelihood(
        chain_token,
        species_token,
        parent_seq,
        infill_range=infill_range,
    )
    ppl = math.exp(-ll)

    print(f"Log likelihood: {round(ll, 3)}")
    print(f"Perplexity: {round(ppl, 3)}")


if __name__ == '__main__':
    main()
