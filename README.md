# Quadratic word embedding models

### Instructions to run.

Recommended: at least 32GB CPU RAM, 16GB GPU RAM.

1. install `jax`, `optax`, `flax`, `datasets`
2. modify *PATH variables in `data/get_core_utils.sh`, then run script to install helper code and set environment variables.
3. optional: paste the bottom section of `data/get_core_utils.sh` into your bashrc. Otherwise, run `get_core_utils.sh` every time before running experiments from a fresh terminal.
4. run `python data/prepare_enwiki.py` (one-time setup, about 2 hours) and `python data/prepare_analogies.py`
5. run `python expts/example.py`

Code flow:
* the `data/*` scripts are for one-time dataset download and setup
* the `expts/*.py` files contain the hyperparameters and birds-eye structure of each experiment
* `qwem.py` contains the logic for the training loop
* `utils.py` defines helper classes for handling hyperparameters, vocabulary, model evaluation, etc.
