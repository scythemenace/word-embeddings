## Overview

The objective of this project was to:
- Train custom word embeddings (Skip-gram and CBOW) using Gensim on the Simple English Wikipedia dataset
- Compare these models with two pretrained embeddings (GloVe and Word2Vec demo)
- Investigate bias using the WEAT metric and the WEFE framework
- Use embeddings as features in a text classification task and compare with a bag-of-words baseline

## Files

- `vocab.txt`: Vocabulary used in the skip-gram model; included for reference.
- `normalize_text_module.py`: Utility module for text preprocessing, including case folding and filtering.  
  > Adapted from my [NormalizeText](https://github.com/scythemenace/NormalizeText) repo with modifications to make it a reusable module for processing any input file.
- `normalize_text_sentence_module.py`: Similar to the above, but designed for sentence-level normalization.
- `w_embeds.py`: Main script for training embeddings, querying vector spaces, running bias evaluation, and performing classification tasks.
- `Report.pdf`: Fully documented report explaining methodology, experimental results, and observations.

## Getting Started

- Make sure Python (any recent version) is installed.
- Required libraries can be installed via pip:
  ```bash
  pip install gensim scikit-learn datasets apache_beam wefe
  ```
