# Quadratic word embedding models

This repo contains the experiments in the paper [Closed-Form Training Dynamics Reveal Learned Features and Linear Structure in Word2Vec-like Models](https://arxiv.org/abs/2502.09863), appearing at NeurIPS 2025.

Here is a [blog post](https://dkarkada.xyz/posts/qwem/) explaining the main results.

### Instructions to run.

Recommended: 32GB CPU RAM, 16GB GPU RAM.

1. install `uv` (if not already in your system) and run `uv sync`.
2. modify *PATH variables in `data/env.sh`, then run script to set environment variables. optional: paste these into your bashrc.
3. run `uv run data/prepare_text8.py` (one-time setup) and `uv run data/prepare_analogies.py`
4. run `uv run expts/example.py` to train a model
5. run `uv run compute_cooccurrence.py text8 10000` to explicitly construct the co-occurrence statistics. Use this to construct M* and factorize in closed form, circumventing the need for gradient descent.

Code flow:
* the scripts in `data/*` are for one-time dataset download and setup
* the files in `expts/*.py` contain the hyperparameters and birds-eye structure of each experiment
* the scripts in `launch/*.sh` launch the figure-generating experiments on a GPU node
* the notebooks `notebooks/*.ipynb` render the figures after the experiments are run and the results are saved
* `qwem.py` contains the logic for the training loop
* `compute_cooccurrence.py` explicitly constructs the co-occurrence matrix for a corpus
* `utils.py` defines helper classes for handling hyperparameters, vocabulary, model evaluation, etc.
* `ExptTrace.py` and `FileManager.py` define more helper classes
* the directory `word2vec_tied/` contains the original SGNS implementation with tied weights. See https://github.com/tmikolov/word2vec/
