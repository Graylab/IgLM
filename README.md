# IgLM
Official repository for IgLM: [Generative Language Modeling for Antibody Design](https://www.biorxiv.org/content/10.1101/2021.12.13.472419v1)

The code and pre-trained models from this work are made available for non-commercial use under the terms of the [JHU Academic Software License Agreement](LICENSE.md).

## Setup
To use IgLM, install dependencies using conda:

```
conda env create -f environment.yml
```

## Usage

### Re-design spans of an antibody sequence
To use IgLM to re-design spans of an antibody sequence, supply the fasta file, the fasta record ID corresponding to the sequence to design, the start index of the span (0-indexed), and the end index of the span (0-indexed, exclusive). 

To generate 100 unique sequences of the anti-tissue factor antibody (1JPT) heavy chain with an IgLM-designed CDR3:
```bash
python infill.py data/antibodies/1jpt/1jpt.fasta :H 98 106 --chain_token [HEAVY] --species_token [HUMAN] --num_seqs 100 
```


### Full antibody sequence generation
IgLM can be used to generate full antibody sequences while conditioning on the chain type and species-of-origin. See Appendix A.5 for starting tokens and sampling temperatures used for the results in the paper.

To generate 100 unique human heavy chain sequences starting with EVQ:
```bash
python full_generation.py --prompt_sequence EVQ --chain_token [HEAVY] --species_token [HUMAN] --num_seqs 100 
```

To generate 100 unique nanobody sequences starting with QVQ:
```bash
python full_generation.py --prompt_sequence QVQ --chain_token [HEAVY] --species_token [CAMEL] --num_seqs 100 
```

### Sequence likelihood calculation
IgLM can be used to calculate the likelihood of a sequence given a chain type and species-of-origin.

Full sequence likelihood calculation:
```bash
python evaluate.py data/antibodies/1jpt/1jpt.fasta :H --chain_token [HEAVY] --species_token [HUMAN]
```

Infilled sequence likelihood calculation:
```bash
python evaluate.py data/antibodies/1jpt/1jpt.fasta :H --start 98 --end 106 --chain_token [HEAVY] --species_token [HUMAN]
```