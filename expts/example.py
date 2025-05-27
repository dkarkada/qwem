import os
from utils import Hyperparams
from qwem import train_embeddings

if os.getenv("DATASETPATH") is None:
    raise ValueError("must set $DATASETPATH environment variable")
if os.getenv("EXPTPATH") is None:
    raise ValueError("must set $EXPTPATH environment variable")

main_dir = os.path.join(os.getenv("EXPTPATH"), "qwem")
expt_name = 'example'

hypers = Hyperparams(
    expt_dir = f'{main_dir}/{expt_name}',
    vocab_sz = 5_000,
    context_len = 16,
    maxsteps = 300_000,
    lr_schedule = None,
    embeddim = 300,
    batch_sz = 20_000,
    checkpt_intervals = None,
    lr = 5e-1,
    init_sz = 1e-1,
    loss = "qwem",
    reweight = "unigram",
    chunk_narticles = 300_000,
    cycle_chunks = True,
)

for dir in [main_dir, hypers.expt_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

train_embeddings(hypers)
