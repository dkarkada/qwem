import os
import urllib.request
import zipfile
from collections import Counter
import numpy as np
from utils import Vocabulary
from FileManager import FileManager

# Text8 dataset is a cleaned version of the first 100MB of Wikipedia
# It contains about 17 million words as a single continuous text file

vocab_sz = 65535    # must be no more than 65535
article_size = 50_000  # words per artificial article. Creates 341 articles

data_dir = os.path.join(os.getenv("DATASETPATH"), "qwem", "text8")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
data_fm = FileManager(data_dir)

# Download and extract text8 dataset if not already present
text8_path = data_fm.get_filename("text8.txt")
if not os.path.exists(text8_path):
    print("Downloading text8 dataset... ", flush=True)
    url = "http://mattmahoney.net/dc/text8.zip"
    zip_path = data_fm.get_filename("text8.zip")
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting text8 dataset... ", flush=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Move the extracted file to our expected location
    extracted_path = os.path.join(data_dir, "text8")
    if os.path.exists(extracted_path):
        os.rename(extracted_path, text8_path)
    
    # Clean up zip file
    os.remove(zip_path)
    print("done.")

# Read and process the text8 file
print("Loading and processing text8 dataset... ", flush=True)
with open(text8_path, 'r') as f:
    text = f.read().strip()
full_corpus = text.split()
print(f"Total words in text8: {len(full_corpus)}")

# Collect unigram statistics, construct vocabulary
print("Collecting word statistics... ", flush=True)
counter = Counter(full_corpus)
counter = {word: c for word, c in counter.items()}
word_counts = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
word_counts = word_counts[:vocab_sz]
data_fm.save(word_counts, "word_counts.pickle")
vocab = Vocabulary(word_counts)

# Create artificial articles of fixed size
print(f"Creating articles of {article_size} words each... ", flush=True)
articles = []
for i in range(0, len(full_corpus), article_size):
    article_words = full_corpus[i:i + article_size]
    articles.append(article_words)

print(f"Created {len(articles)} articles")


# Construct bin file
print(f"Creating bin file with {len(articles)} articles (vocab size = {vocab_sz})... ", flush=True)
filename = data_fm.get_filename("corpus.bin")
arr_len_upperbound = len(full_corpus) + len(articles)
arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len_upperbound,))
EOF_token = vocab_sz
assert EOF_token >= len(vocab.words)
assert EOF_token < 2**16
idx = 0
article_arr_idxs = []
for article in articles:
    article_arr_idxs.append(idx)
    article_tokens = [vocab.word2token[word] for word in article if word in vocab.word2token]
    article_tokens.append(EOF_token)
    arr[idx:idx + len(article_tokens)] = article_tokens
    idx += len(article_tokens)
article_arr_idxs.append(idx)
arr.flush()
with open(filename, 'r+b') as f:
    f.truncate(idx * np.dtype(np.uint16).itemsize)
data_fm.save(np.array(article_arr_idxs), "article_arr_idxs.npy")

print("done.")
