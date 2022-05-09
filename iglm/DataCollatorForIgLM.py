import random

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, ClassVar

import itertools

import torch
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
import numpy as np


@dataclass
class DataCollatorForIgLM:
    """
    Data collator used for training Immunoglobulin Language ModeL (IgLM).
    Inputs are dynamically padded to the maximum length of a batch if they are not all of the same length.

    - random_masking: whether to randomly mask when collating inputs. Used for training.
    - mask_mode: one of ['random', 'cdr1', 'cdr2', 'cdr3', 'cdr1_graft', 'cdr2_graft', 'cdr3_graft'] for how to mask.
       - 'random': min_mask_len and max_mask_len for uniform random mask lengths
       - '*_graft': host_example should be a BertTokenized dict with key 'input_ids'
    - infilled_only_loss: compute loss only for infilled portions (i.e. all tokens after [SEP], including [CLS])
    - remove_unused_columns: if True, remove all columns not in ['input_ids', 'labels'] for Trainer.train().
    """
    SPECIES_TO_TOKEN: ClassVar[dict] = {
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
    CHAIN_TO_TOKEN: ClassVar[dict] = {
        "Heavy": "[HEAVY]",
        "Light": "[LIGHT]"
    }
    SAP_TO_TOKEN: ClassVar[dict] = {
        "SAP_LOW": "[SAP_LOW]",
        "SAP_MID": "[SAP_MID]",
        "SAP_HIGH": "[SAP_HIGH]"
    }
    SOLUBILITY_TO_TOKEN: ClassVar[dict] = {
        "SOL_LOW": "[SOL_LOW]",
        "SOL_MID": "[SOL_MID]",
        "SOL_HIGH": "[SOL_HIGH]"
    }
    NULL_CONDITION_TOKEN: ClassVar[str] = '[NULL]'

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None
    condition_on_chain: bool = False
    condition_on_species: bool = False
    condition_on_sap_and_solubility: bool = False
    single_token_conditioning: bool = False
    null_conditioning: bool = False
    infilled_only_loss: bool = False
    include_answer: bool = True
    remove_unused_columns: bool = True

    mask_mode: str = 'random'
    min_mask_len: int = 10
    max_mask_len: int = 20
    host_example: dict = None

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            input_ids, batch = self.build_iglm_examples(examples)
            batched_input_ids = self.tokenizer.pad(input_ids, return_tensors="pt",
                                                   pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            assert NotImplementedError

        # Update batch with padded input_ids and attention_mask
        batch.update(batched_input_ids)

        labels = batch["input_ids"].clone()

        # Compute loss only for tokens after [SEP], including [CLS]
        if self.infilled_only_loss:
            sep_token_mask = (labels == self.tokenizer.sep_token_id)
            mask = torch.cumsum(sep_token_mask, dim=1)
            mask = mask.bool()
            mask = mask & (~sep_token_mask)
            labels[~mask] = -100

        # Mask out pad tokens
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch

    def build_iglm_examples(self, examples):
        """
        Build IgLM examples based on self.mask_mode. Modifies examples directly, but also returns them.
        """
        for example in examples:
            # Choose start and mask_len based on self.mask_mode
            start, mask_len = None, None
            if self.mask_mode == 'random':
                start, mask_len = self._get_rand_mask(example)
            elif self.mask_mode[:3] == 'cdr':
                start, mask_len = self._get_cdr_mask(example)
            else:
                assert NotImplementedError

            if self.mask_mode[5:] == 'graft':
                self._build_graft_example(example, start, mask_len)
            else:
                self._build_iglm_example(example, start, mask_len)

            # Delete columns except for those that appear in model signature
            if self.remove_unused_columns:
                input_ids = example['input_ids']
                example.clear()
                example['input_ids'] = input_ids

        batch = {key: [example[key] for example in examples] for key in examples[0].keys()}
        input_ids = {"input_ids": batch["input_ids"]}
        return input_ids, batch

    def _build_iglm_example(self, example, start, mask_len):
        """
        Builds IgLM task from input_ids in example by masking out mask_len tokens, starting from start.

        start - 0-indexed position of sequence (including original CLS token)
        """
        seq = example['input_ids']
        end = start + mask_len

        # Replace tokens with [MASK] id and add span to end
        iglm_seq = []
        if self.condition_on_chain:
            chain_token = self.CHAIN_TO_TOKEN[example['chain_type']]
            iglm_seq += [self.tokenizer.convert_tokens_to_ids(chain_token)]
        if self.condition_on_species:
            species_token = self.SPECIES_TO_TOKEN[example['species']]
            iglm_seq += [self.tokenizer.convert_tokens_to_ids(species_token)]
        if self.condition_on_sap_and_solubility and not self.null_conditioning:
            if example['sap'] is not None and example['solubility'] is not None:
                sap_token = self.SAP_TO_TOKEN[example['sap']]
                solubility_token = self.SOLUBILITY_TO_TOKEN[example['solubility']]

                if not self.single_token_conditioning:
                    iglm_seq += [self.tokenizer.convert_tokens_to_ids(sap_token)]
                    iglm_seq += [self.tokenizer.convert_tokens_to_ids(solubility_token)]
                else:
                    p = np.random.uniform()
                    if p < 1 / 3:
                        iglm_seq += [self.tokenizer.convert_tokens_to_ids(sap_token)]
                    elif p < 2 / 3:
                        iglm_seq += [self.tokenizer.convert_tokens_to_ids(solubility_token)]
                    else:
                        iglm_seq += [self.tokenizer.convert_tokens_to_ids(sap_token)]
                        iglm_seq += [self.tokenizer.convert_tokens_to_ids(solubility_token)]
            else:
                assert (example['sap'] is None) and (example['solubility'] is None), \
                    "one of solubility or sap is None, but not both"

        if self.condition_on_sap_and_solubility and self.null_conditioning:
            if example['sap'] is not None and example['solubility'] is not None:
                # if labels exist, 2/5 of the time include one conditioning token and 3/5 include both
                sap_token = self.SAP_TO_TOKEN[example['sap']]
                solubility_token = self.SOLUBILITY_TO_TOKEN[example['solubility']]

                p = np.random.uniform()
                if p < 1 / 5:
                    iglm_seq += [self.tokenizer.convert_tokens_to_ids(sap_token)]
                    iglm_seq += [self.tokenizer.convert_tokens_to_ids(self.NULL_CONDITION_TOKEN)]
                elif p < 2 / 5:
                    iglm_seq += [self.tokenizer.convert_tokens_to_ids(self.NULL_CONDITION_TOKEN)]
                    iglm_seq += [self.tokenizer.convert_tokens_to_ids(solubility_token)]
                else:
                    iglm_seq += [self.tokenizer.convert_tokens_to_ids(sap_token)]
                    iglm_seq += [self.tokenizer.convert_tokens_to_ids(solubility_token)]

            elif (example['sap'] is None) and (example['solubility'] is None):
                # if labels both don't exist, set to null
                iglm_seq += [self.tokenizer.convert_tokens_to_ids(self.NULL_CONDITION_TOKEN)]
                iglm_seq += [self.tokenizer.convert_tokens_to_ids(self.NULL_CONDITION_TOKEN)]
            else:
                assert False, "one of solubility or sap is None, but not both"

        if self.include_answer:
            iglm_seq += itertools.chain(seq[1:start], [self.tokenizer.mask_token_id], seq[end:],  # SEP already included
                                       seq[start:end], [self.tokenizer.cls_token_id])
        else:
            iglm_seq += itertools.chain(seq[1:start], [self.tokenizer.mask_token_id], seq[end:])
            example['masked_segment'] = seq[start:end]  # Store masked portion for later

        example['input_ids'] = iglm_seq
        example['mask_len'] = mask_len  # Useful to have
        return example

    def _build_graft_example(self, example, start, mask_len):
        """
        Use self.host_example as the framework and graft example loop onto it.
        """
        host_start, host_mask_len = self._get_cdr_mask(self.host_example)
        host_seq = self.host_example['input_ids']
        seq = example['input_ids']
        host_end = host_start + host_mask_len
        end = start + mask_len

        iglm_seq = []
        if self.condition_on_chain:
            host_chain_token = self.CHAIN_TO_TOKEN[self.host_example['chain_type']]
            iglm_seq += [self.tokenizer.convert_tokens_to_ids(host_chain_token)]
        if self.condition_on_species:
            host_species_token = self.SPECIES_TO_TOKEN[self.host_example['species']]
            iglm_seq += [self.tokenizer.convert_tokens_to_ids(host_species_token)]

        iglm_seq += itertools.chain(host_seq[1:host_start], [self.tokenizer.mask_token_id], host_seq[host_end:],
                                   seq[start:end], [self.tokenizer.cls_token_id])

        example['host_segment'] = host_seq[host_start:host_end]
        example['grafted_segment'] = seq[start:end]
        example['input_ids'] = iglm_seq
        return example

    def _get_cdr_mask(self, example):
        """
        Get starting point and mask len based on CDR definition. CDR definitions should be in the form
        '[25:32]' inclusive, 0-indexed (not including CLS token).
        Returns a (start, mask_len)

        - loop should be one of ['cdr1', 'cdr2', 'cdr3']
        """
        loop = self.mask_mode[:4]
        start, end = example[loop].lstrip('[').rstrip(']').split(':')
        start, end = int(start) + 1, int(end) + 1  # Account for CLS token in input_ids
        mask_len = end - start + 1  # inclusive interval

        return start, mask_len

    def _get_rand_mask(self, example: List[dict]):
        """
        Choose a random starting point and mask len (as specified by self.min_mask_len and self.max_mask_len).
        Returns a (start, mask_len) tuple.
        """
        seq = example['input_ids']
        mask_len = random.randint(self.min_mask_len, self.max_mask_len)
        start = random.randint(1, len(seq) - mask_len - 1)

        return start, mask_len