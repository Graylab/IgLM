import argparse
import os

import torch
import transformers

from iglm.infill_utils import BAD_WORD_IDS, iglm_generate, CHAIN_TO_TOKEN, SPECIES_TO_TOKEN
from util import set_seeds
from datetime import datetime


def main():
    """
    Generate antibody sequences given chain and species tokens.
    """
    now = str(datetime.now().strftime('%y-%m-%d %H:%M:%S'))

    parser = argparse.ArgumentParser(description='Generate antibody sequences from scratch')
    parser.add_argument('chkpt_dir', type=str, help='Model checkpoint directory')
    parser.add_argument('--prompt_tokens', type=str, help='Prompt tokens')
    parser.add_argument('--chain', type=str, default='heavy', help='Chain to design (i.e. "heavy" or "light")')
    parser.add_argument('--species', type=str, default='human', help='Species')
    parser.add_argument('--num_seqs', type=int, default=1000, help='Number of sequences to generate')
    parser.add_argument('--temperature', type=float, default=1, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=1, help='p for top-p sampling')
    parser.add_argument('--output_dir', type=str, default=f'output_dir/full_generation_{now}')
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

    # Get starting tokens
    chain_token = CHAIN_TO_TOKEN[args.chain.capitalize()]
    species_token = SPECIES_TO_TOKEN[args.species]

    prompt_tokens = list(args.prompt_tokens)
    start_tokens = [chain_token, species_token]
    start_tokens += prompt_tokens
    print(f"Starting tokens: {start_tokens}")
    start_tokens = torch.Tensor([tokenizer.convert_tokens_to_ids(start_tokens)]).int().to(device)

    # Generate designed seqs
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
