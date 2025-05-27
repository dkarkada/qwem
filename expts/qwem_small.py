import os
from utils import Hyperparams
from qwem import train_embeddings

if os.getenv("DATASETPATH") is None:
    raise ValueError("must set $DATASETPATH environment variable")
if os.getenv("EXPTPATH") is None:
    raise ValueError("must set $EXPTPATH environment variable")

main_dir = os.path.join(os.getenv("EXPTPATH"), "qwem")
expt_name = 'qwem-small'

hypers = Hyperparams(
    expt_dir = f'{main_dir}/{expt_name}',
    vocab_sz = 10_000,
    context_len = 16,
    maxsteps = 4_000_000,
    lr_schedule = {3_000_000: 0.1},
    embeddim = 200,
    batch_sz = 50_000,
    checkpt_intervals = [(0, 1e4, 10), (1e4, 1e5, 9),
                         (1e5, 5e5, 40), (5e5, 4e6, 8)],
    lr = 5,
    init_sz = 1e-12,
    loss = "qwem",
    reweight = "omniscient",
    chunk_narticles = 1586650,
    cycle_chunks = False,
)

for dir in [main_dir, hypers.expt_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

train_embeddings(hypers)
