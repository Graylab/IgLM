# IgLM
Official repository for IgLM: [Generative Language Modeling for Antibody Design](https://www.biorxiv.org/content/10.1101/2021.12.13.472419v1)

## Setup
To use IgLM, install dependencies using conda:

```
conda env create -f environment.yml
```

## Usage

### Re-design spans of an antibody sequence
To use IgLM to re-design spans of an antibody sequence, supply the fasta file, the fasta record ID corresponding to the sequence to design, the start index of the span (0-indexed), and the end index of the span (0-indexed, exclusive). 

To generate 100 unique sequences of the anti-tissue factor antibody (1JPT) heavy chain with an IgLM-designed CDR3:
```
python infill.py data/antibodies/1jpt/1jpt.fasta :H 98 106 --chain_token [HEAVY] --species_token [HUMAN] --num_seqs 100 
```


### Full antibody sequence generation
IgLM can be used to generate full antibody sequences while conditioning on the chain type and species-of-origin. See Appendix A.5 for starting tokens and sampling temperatures used for the results in the paper.

To generate 100 unique human heavy chain sequences starting with EVQ:
```
python full_generation.py --prompt_tokens EVQ --chain_token [HEAVY] --species_token [HUMAN] --num_seqs 100 
```

To generate 100 unique nanobody sequences starting with QVQ:
```
python full_generation.py --prompt_tokens QVQ --chain_token [HEAVY] --species_token [CAMEL] --num_seqs 100 
```
