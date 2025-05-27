import os
from utils import Hyperparams
from qwem import train_embeddings

if os.getenv("DATASETPATH") is None:
    raise ValueError("must set $DATASETPATH environment variable")
if os.getenv("EXPTPATH") is None:
    raise ValueError("must set $EXPTPATH environment variable")

main_dir = os.path.join(os.getenv("EXPTPATH"), "qwem")
expt_name = 'sgns-fast'

hypers = Hyperparams(
    expt_dir = f'{main_dir}/{expt_name}',
    vocab_sz = 10_000,
    context_len = 16,
    maxsteps = 50_000,
    lr_schedule = None,
    embeddim = 200,
    batch_sz = 50_000,
    checkpt_intervals = None,
    lr = 15,
    init_sz = 3e-1,
    loss = "sgns",
    reweight = "sgns",
    chunk_narticles = 1586650,
    cycle_chunks = False,
)

for dir in [main_dir, hypers.expt_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

train_embeddings(hypers)
