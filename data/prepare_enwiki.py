from datasets import load_dataset, load_from_disk
import os
from collections import Counter
from tqdm import tqdm
import numpy as np
from utils import Vocabulary
from FileManager import FileManager

# Full wikipedia is 6407814 articles
# In 200k articles: mean article length is 678, max is 50k
#                   [0.5, 0.9, 0.95, 0.99] quantiles = [364, 1470, 2233, 5362.01]

vocab_sz = 20000    # must be less than 65535
min_length = 500
dirname = "min500"

data_dir = os.path.join(os.getenv("DATASETPATH"), "qwem")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
data_fm = FileManager(data_dir)

try:
    cleaned_ds = load_from_disk(data_fm.get_filename("cleaned_wikipedia"))
except:
    ds = load_dataset("wikimedia/wikipedia", "20231101.en")
    train_ds = ds["train"]

    print("Cleaning dataset... ", flush=True)
    def process_document(article):
        import re
        cleaned = re.findall(r"[a-z]+", article["text"].lower())
        return {"cleaned": cleaned}

    cleaned_ds = train_ds.map(process_document, num_proc=8)
    cleaned_ds.save_to_disk(data_fm.get_filename("cleaned_wikipedia"))
    print("done.")

data_fm.set_filepath(dirname)
# Collect unigram statistics, construct vocabulary
counter = Counter()
article_idxs = []
arr_len_upperbound = 0
for i in tqdm(range(len(cleaned_ds))):
    article = cleaned_ds[i]["cleaned"]
    if len(article) >= min_length:
        article_idxs.append(i)
        arr_len_upperbound += 1 + len(article)
        counter.update(article)
counter = {word: c for word, c in counter.items()}

word_counts = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
word_counts = word_counts[:vocab_sz]
data_fm.save(word_counts, "word_counts.pickle")
vocab = Vocabulary(word_counts)

# Construct bin file
print(f"Using {100*len(article_idxs)/6407814:0.2f}% of articles, min length {min_length}.", flush=True)
print(f"Creating bin file (vocab size = {vocab_sz})... ", flush=True)
filename = data_fm.get_filename("enwiki.bin")
arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len_upperbound,))
EOF_token = vocab_sz
assert EOF_token >= len(vocab.words)
assert EOF_token < 2**16
idx = 0
article_arr_idxs = []
for i in tqdm(article_idxs):
    article = cleaned_ds[i]["cleaned"]
    assert len(article) >= min_length
    corpus = np.array([vocab.word2token[word] for word in article
                       if word in vocab.word2token] + [EOF_token],
                      dtype=np.uint16)
    assert idx + len(corpus) <= arr_len_upperbound
    arr[idx : idx + len(corpus)] = corpus
    article_arr_idxs.append(idx)
    idx += len(corpus)
article_arr_idxs.append(idx)
arr.flush()
with open(filename, 'r+b') as f:
    f.truncate(idx * np.dtype(np.uint16).itemsize)
data_fm.save(np.array(article_arr_idxs), "article_arr_idxs.npy")

print("done.")
