import os
from utils import Hyperparams
from qwem import train_embeddings

if os.getenv("DATASETPATH") is None:
    raise ValueError("must set $DATASETPATH environment variable")
if os.getenv("EXPTPATH") is None:
    raise ValueError("must set $EXPTPATH environment variable")

main_dir = os.path.join(os.getenv("EXPTPATH"), "qwem")
expt_name = 'sgns-cocanow'

hypers = Hyperparams(
    expt_dir = f'{main_dir}/{expt_name}',
    vocab_sz = 30_000,
    context_len = 16,
    maxsteps = 250_000,
    lr_schedule = {25_000: 0.25, 40_000: 0.9, 50_000: 0.9, 60_000: 0.9,
                   70_000: 0.9, 80_000: 0.9, 90_000: 0.9, 100_000: 0.9,
                   150_000: 0.5},
    embeddim = 200,
    batch_sz = 50_000,
    checkpt_intervals = [(0, 250_000, 6)],
    lr = 100,
    init_sz = 3e-2,
    ns_weight = 2.0,
    loss = "sgns",
    reweight = "sgns",
    dataset = "cocanow",
    chunk_narticles = 818,
    cycle_chunks = False,
)

for dir in [main_dir, hypers.expt_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

train_embeddings(hypers)
