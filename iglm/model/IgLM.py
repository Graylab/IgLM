import os

import torch
import transformers
from tqdm import tqdm

import iglm
from iglm.model.tokens import *
from iglm.model.utils import *
from iglm.utils.general import exists

project_path = os.path.dirname(os.path.realpath(iglm.__file__))
DEFAULT_CHKPT_DIR = os.path.join(project_path, os.path.pardir,
                                 'trained_models/IgLM')
VOCAB_FILE = os.path.join(project_path, os.path.pardir, 'vocab.txt')


class IgLM():

    def __init__(self, chkpt_dir: str = None):
        if not exists(chkpt_dir):
            chkpt_dir = DEFAULT_CHKPT_DIR

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = transformers.GPT2LMHeadModel.from_pretrained(
            chkpt_dir).to(self.device)
        self.model.eval()

        self.tokenizer = transformers.BertTokenizerFast(vocab_file=VOCAB_FILE,
                                                        do_lower_case=False)

    def _generate(self, starting_tokens, num_to_generate, top_p, temperature):
        decoded_seqs = set()  # Set to remove duplicates
        pbar = tqdm(total=num_to_generate)
        while len(decoded_seqs) < num_to_generate:
            seq = self.model.generate(
                starting_tokens,
                max_length=150,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.cls_token_id,
                forced_eos_token_id=self.tokenizer.cls_token_id,
                bad_words_ids=BAD_WORD_IDS,
                do_sample=True,
                top_p=top_p,
                temperature=temperature).detach().cpu().numpy()

            seq = seq[0]  # Squeeze out batch dimension
            if validate_generated_seq(seq, self.tokenizer):
                decoded_tokens = self.tokenizer.decode(
                    iglm_to_infilled(seq, self.tokenizer))
                decoded_seq = ''.join(decoded_tokens).replace(' ', '')
                if decoded_seq not in decoded_seqs:
                    decoded_seqs.add(decoded_seq)
                    pbar.update(1)

        pbar.close()
        return list(decoded_seqs)

    def generate(self,
                 chain_token,
                 species_token,
                 prompt_sequence,
                 num_to_generate=1000,
                 top_p=1,
                 temperature=1):
        start_tokens = [chain_token, species_token]

        if exists(prompt_sequence):
            prompt_tokens = list(prompt_sequence)
            start_tokens += prompt_tokens

        start_tokens = torch.Tensor([
            self.tokenizer.convert_tokens_to_ids(start_tokens)
        ]).int().to(self.device)

        assert (start_tokens != self.tokenizer.unk_token_id
                ).all(), "Unrecognized token supplied in starting tokens"

        generated_seqs = self._generate(
            start_tokens,
            num_to_generate=num_to_generate,
            top_p=top_p,
            temperature=temperature,
        )

        return generated_seqs

    def infill(
        self,
        chain_token,
        species_token,
        sequence,
        infill_range,
        num_to_generate=1000,
        top_p=1,
        temperature=1,
    ):
        masked_seq = mask_span(
            sequence,
            infill_range[0],
            infill_range[1],
        )  # mask using provided indices
        start_tokens = [chain_token, species_token] + masked_seq
        start_tokens = torch.Tensor([
            self.tokenizer.convert_tokens_to_ids(start_tokens)
        ]).int().to(self.device)

        assert (start_tokens != self.tokenizer.unk_token_id
                ).all(), "Unrecognized token supplied in starting tokens"

        generated_seqs = self._generate(
            start_tokens,
            num_to_generate=num_to_generate,
            top_p=top_p,
            temperature=temperature,
        )

        return generated_seqs

    def log_likelihood(
        self,
        chain_token,
        species_token,
        sequence,
        infill_range=None,
    ):
        if exists(infill_range):
            sequence = mask_span(
                sequence,
                infill_range[0],
                infill_range[1],
                append_span=True,
            )  # mask using provided indices

        token_seq = [chain_token, species_token] + sequence

        if exists(infill_range):
            token_seq += [self.tokenizer.cls_token]
        else:
            token_seq += [self.tokenizer.sep_token]

        token_seq = torch.Tensor([
            self.tokenizer.convert_tokens_to_ids(token_seq)
        ]).int().to(self.device)

        assert (token_seq != self.tokenizer.unk_token_id
                ).all(), "Unrecognized token supplied in starting tokens"

        if exists(infill_range):
            eval_start = np.nonzero(
                token_seq[0] == self.tokenizer.sep_token_id)[0].item()
        else:
            eval_start = 1

        logits = self.model(token_seq).logits
        shift_logits = logits[..., eval_start:-1, :].contiguous()
        shift_labels = token_seq[..., eval_start + 1:].contiguous().long()
        nll = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean',
        )

        return -nll.item()