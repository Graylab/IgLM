# IgLM
Official repository for IgLM: [Generative Language Modeling for Antibody Design](https://www.biorxiv.org/content/10.1101/2021.12.13.472419v1)

The code and pre-trained models from this work are made available for non-commercial use under the terms of the [JHU Academic Software License Agreement](LICENSE.md).

## Setup
To use IgLM, install via pip:
```bash
pip install iglm
```

Alternatively, you can clone this repository and install the package locally:
```bash
$ git clone git@github.com:Graylab/IgLM.git 
$ pip install IgFold
```

## Command line usage

IgLM supports sequence infilling, sequence generation (with prompting), and sequence evaluation from the command line.

### Re-design spans of an antibody sequence
To use IgLM to re-design spans of an antibody sequence, supply the fasta file, the fasta record ID corresponding to the sequence to design, the start index of the span (0-indexed), and the end index of the span (0-indexed, exclusive). 

To generate 100 unique sequences of the anti-tissue factor antibody (1JPT) heavy chain with an IgLM-designed CDR3:
```bash
iglm_infill data/antibodies/1jpt/1jpt.fasta :H 98 106 --chain_token [HEAVY] --species_token [HUMAN] --num_seqs 100 
```


### Full antibody sequence generation
IgLM can be used to generate full antibody sequences while conditioning on the chain type and species-of-origin. See Appendix A.5 for starting tokens and sampling temperatures used for the results in the paper.

To generate 100 unique human heavy chain sequences starting with EVQ:
```bash
iglm_generate --prompt_sequence EVQ --chain_token [HEAVY] --species_token [HUMAN] --num_seqs 100 
```

To generate 100 unique nanobody sequences starting with QVQ:
```bash
iglm_generate --prompt_sequence QVQ --chain_token [HEAVY] --species_token [CAMEL] --num_seqs 100 
```

### Sequence evaluation
IgLM can be used to calculate the log likelihood of a sequence given a chain type and species-of-origin.

Full sequence log likelihood calculation:
```bash
iglm_evaluate data/antibodies/1jpt/1jpt.fasta :H --chain_token [HEAVY] --species_token [HUMAN]
```

Infilled sequence log likelihood calculation:
```bash
iglm_evaluate data/antibodies/1jpt/1jpt.fasta :H --start 98 --end 106 --chain_token [HEAVY] --species_token [HUMAN]
```

## Package usage

IgLM may also be used as a Python package, enabling the above use cases and more flexible usage.

### Re-design spans of an antibody sequence
To use IgLM to re-design spans of an antibody sequence, supply the fasta file, the fasta record ID corresponding to the sequence to design, the start index of the span (0-indexed), and the end index of the span (0-indexed, exclusive). 

To generate 100 unique sequences of the anti-tissue factor antibody (1JPT) heavy chain with an IgLM-designed CDR3:
```python
from iglm import IgLM

iglm = IgLM()

parent_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKEYYMHWVRQAPGKGLEWVGLIDPEQGNTIYDPKFQDRATISADNSKNTAYLQMNSLRAEDTAVYYCARDTAAYFDYWGQGTLVTVS"
chain_token = "[HEAVY]"
species_token = "[HUMAN]"
infill_range = (98, 106)
num_seqs = 100

generated_seqs = iglm.infill(
    parent_sequence,
    chain_token,
    species_token,
    infill_range=infill_range,
    num_to_generate=num_seqs,
)
```


### Full antibody sequence generation
IgLM can be used to generate full antibody sequences while conditioning on the chain type and species-of-origin. See Appendix A.5 for starting tokens and sampling temperatures used for the results in the paper.

To generate 100 unique human heavy chain sequences starting with EVQ:
```python
from iglm import IgLM

iglm = IgLM()

prompt_sequence = "EVQ"
chain_token = "[HEAVY]"
species_token = "[HUMAN]"
num_seqs = 100

generated_seqs = iglm.generate(
    chain_token,
    species_token,
    prompt_sequence=prompt_sequence,
    num_to_generate=num_seqs,
)
```

To generate 100 unique nanobody sequences starting with QVQ:
```python
from iglm import IgLM

iglm = IgLM()

prompt_sequence = "QVQ"
chain_token = "[HEAVY]"
species_token = "[CAMEL]"
num_seqs = 100

generated_seqs = iglm.generate(
    chain_token,
    species_token,
    prompt_sequence=prompt_sequence,
    num_to_generate=num_seqs,
)
```

### Sequence evaluation
IgLM can be used to calculate the log likelihood of a sequence given a chain type and species-of-origin.

Full sequence log likelihood calculation:
```python
import math
from iglm import IgLM

iglm = IgLM()

sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKEYYMHWVRQAPGKGLEWVGLIDPEQGNTIYDPKFQDRATISADNSKNTAYLQMNSLRAEDTAVYYCARDTAAYFDYWGQGTLVTVS"
chain_token = "[HEAVY]"
species_token = "[HUMAN]"

log_likelihood = iglm.log_likelihood(
    sequence,
    chain_token,
    species_token,
    infill_range=infill_range,
)
perplexity = math.exp(-log_likelihood)
```

Infilled sequence log likelihood calculation:
```python
import math
from iglm import IgLM

iglm = IgLM()

sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKEYYMHWVRQAPGKGLEWVGLIDPEQGNTIYDPKFQDRATISADNSKNTAYLQMNSLRAEDTAVYYCARDTAAYFDYWGQGTLVTVS"
chain_token = "[HEAVY]"
species_token = "[HUMAN]"
infill_range = (98, 106)

log_likelihood = iglm.log_likelihood(
    sequence,
    chain_token,
    species_token,
    infill_range=infill_range,
)
perplexity = math.exp(-log_likelihood)
```