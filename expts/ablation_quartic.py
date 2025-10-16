import os
from utils import Hyperparams
from qwem import train_embeddings

if os.getenv("DATASETPATH") is None:
    raise ValueError("must set $DATASETPATH environment variable")
if os.getenv("EXPTPATH") is None:
    raise ValueError("must set $EXPTPATH environment variable")

main_dir = os.path.join(os.getenv("EXPTPATH"), "qwem")
expt_name = 'ablation-quartic'

hypers = Hyperparams(
    expt_dir = f'{main_dir}/{expt_name}',
    vocab_sz = 20_000,
    context_len = 16,
    maxsteps = 5_000_000,
    lr_schedule = None,
    embeddim = 150,
    batch_sz = 50_000,
    checkpt_intervals = [(0, 5_000_000, 6)],
    lr = 2,
    init_sz = 1e-3,
    ns_weight = 2.0,
    loss = "qwem",
    reweight = "sgns",
    dataset = "cocanow",
    chunk_narticles = 818,
    cycle_chunks = False,
)

for dir in [main_dir, hypers.expt_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

train_embeddings(hypers)
