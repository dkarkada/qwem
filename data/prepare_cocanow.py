import os
import urllib.request
import zipfile
from collections import Counter
import re
import numpy as np
from utils import Vocabulary
from FileManager import FileManager

# we concatenate the following datasets
# COCA (Corpus of Contemporary American English) dataset
# NOW (News on the Web) dataset

vocab_sz = 65535    # must be no more than 65535
article_size = 50_000  # words per artificial article. Creates 818 articles

data_dir = os.path.join(os.getenv("DATASETPATH"), "qwem", "cocanow")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
data_fm = FileManager(data_dir)

def preprocess_text(text):
    """
    Preprocess text following text8 methodology:
    - Convert to lowercase
    - Convert numbers to spelled out form
    - Keep only a-z characters, convert everything else to spaces
    - Normalize whitespace
    """
    # Convert to lowercase
    text = text.lower()
    
    # Convert numbers to spelled out form
    digit_map = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    
    # Replace each digit with its word form
    for digit, word in digit_map.items():
        text = text.replace(digit, f' {word} ')
    
    # Keep only a-z characters, convert everything else to single space
    text = re.sub(r'[^a-z]+', ' ', text)
    
    # Normalize whitespace (remove extra spaces)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


# Download and extract COCA + NOW datasets if not already present
cocanow_path = data_fm.get_filename("cocanow.txt")
if not os.path.exists(cocanow_path):
    coca_path = data_fm.get_filename("coca.txt")
    print("Downloading COCA dataset... ", flush=True)
    url = "https://www.corpusdata.org/coca/samples/coca-samples-text.zip"
    zip_path = data_fm.get_filename("coca.zip")

    try:
        urllib.request.urlretrieve(url, zip_path)

        print("Extracting COCA dataset... ", flush=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        # Process extracted files - COCA typically contains multiple text files
        # Combine them into a single text file
        combined_text = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.txt') and file != 'coca.txt':
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().strip()
                        # Basic text cleaning
                        words = content.split()
                        combined_text.extend(words)

        # Write combined text to coca.txt
        coca_txt = ' '.join(combined_text)
        print("Preprocessing... ", flush=True)
        coca_txt = preprocess_text(coca_txt)
        with open(coca_path, 'w', encoding='utf-8') as f:
            f.write(coca_txt)

        # Clean up zip file
        os.remove(zip_path)
        print("done.")
        
    except Exception as e:
        print(f"Error downloading COCA: {e}")
        exit(1)

    now_path = data_fm.get_filename("now.txt")
    print("Downloading NOW dataset... ", flush=True)
    url = "https://www.corpusdata.org/now/samples/now-text-2024.zip"
    zip_path = data_fm.get_filename("now.zip")

    try:
        urllib.request.urlretrieve(url, zip_path)

        print("Extracting NOW dataset... ", flush=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        # Process extracted files - NOW typically contains multiple text files
        # Combine them into a single text file
        combined_text = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.txt') and file not in ['coca.txt', 'now.txt', 'cocanow.txt']:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().strip()
                        # Basic text cleaning
                        words = content.split()
                        combined_text.extend(words)

        # Write combined text to now.txt
        now_txt = ' '.join(combined_text)
        print("Preprocessing... ", flush=True)
        now_txt = preprocess_text(now_txt)
        with open(now_path, 'w', encoding='utf-8') as f:
            f.write(now_txt)

        # Clean up zip file
        os.remove(zip_path)
        print("done.")
        
    except Exception as e:
        print(f"Error downloading NOW: {e}")
        exit(1)

    coca_corpus = coca_txt.split()
    print(f"{len(coca_corpus)/1e6:.2f}M words in COCA", flush=True)
    now_corpus = now_txt.split()
    print(f"{len(now_corpus)/1e6:.2f}M words in NOW", flush=True)
    full_corpus = coca_corpus + now_corpus
    print(f"Total: {len(full_corpus)/1e6:.2f}M words", flush=True)

    # Save the combined corpus
    cocanow_path = data_fm.get_filename("cocanow.txt")
    full_text = ' '.join(full_corpus)
    with open(cocanow_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

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
