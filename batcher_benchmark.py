import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import os
import utils
from FileManager import FileManager

from qwem import get_batch_generator


H = utils.Hyperparams(
    expt_dir='',
    vocab_sz = 1000,
    context_len = 32,
    batch_sz = 20000,
    chunk_narticles = 10000
)

data_dir = os.path.join(os.getenv("DATASETPATH"), "qwem")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
data_fm = FileManager(data_dir)
data_fm.set_filepath("min500")

word_counts = data_fm.load("word_counts.pickle")
assert len(word_counts) >= H.vocab_sz
word_counts = word_counts[:H.vocab_sz]
vocab = utils.Vocabulary(word_counts)
unigram = vocab.counts / vocab.counts.sum()
pos_distr = 1 / unigram
neg_distr = 1 / unigram
q_pos = lambda i,j: 1/(pos_distr[i] * pos_distr[j])
q_neg = lambda i,j: 1/(neg_distr[i] * neg_distr[j])

print("Warming up...")
batch_gen = get_batch_generator(H, data_fm, q_pos, q_neg)
next(batch_gen)
next(batch_gen)

print("Starting benchmark...")
times = []
ntrials = 100
for _ in range(ntrials):
    start = time.perf_counter()
    for i in range(10):
        batch = next(batch_gen)
    end = time.perf_counter()
    times.append((end - start) / 10)

avg = np.mean(times)
std = np.std(times)
print(f"avg_time_per_{H.batch_sz//1000}k = {avg:.6f}s ± {std:.6f}s")
