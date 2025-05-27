import numpy as np
from numba import njit
from tqdm import tqdm
import os

from FileManager import FileManager

data_dir = os.path.join(os.getenv("DATASETPATH"), "qwem")
data_fm = FileManager(data_dir)
data_fm.set_filepath("min500")

context_len = 16
dtype = np.uint16
corpus_vocabsz = len(data_fm.load("word_counts.pickle"))
Cij = np.zeros((corpus_vocabsz, corpus_vocabsz), dtype=np.float32)
Pij = np.zeros((corpus_vocabsz, corpus_vocabsz), dtype=np.float32)
article_idxs = data_fm.load("article_arr_idxs.npy")
corpus_fn = data_fm.get_filename("enwiki.bin")

@njit
def update_cooccurrence(Cij, Pij, article):
    for i in range(0, len(article) - (context_len+1)):
        for j in range(1, context_len+1):
            w, v = article[i], article[i+j]
            counts = context_len+1-j
            Cij[w, v] += counts
            Cij[v, w] += counts
            Pij[w, v] += 1
            Pij[v, w] += 1

corpus = np.memmap(corpus_fn, dtype=dtype, mode='r')
for i in tqdm(range(len(article_idxs) - 1)):
    start, stop = article_idxs[i], article_idxs[i+1] - 1
    article = corpus[start:stop]
    update_cooccurrence(Cij, Pij, article)

corpus_stats = {
    "counts": Pij,
    "counts_reweight": Cij,
    "context_len": context_len,
}
data_fm.save(corpus_stats, "corpus_stats.pickle")
