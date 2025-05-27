import os
import requests
from FileManager import FileManager

data_dir = os.path.join(os.getenv("DATASETPATH"), "qwem")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
data_fm = FileManager(data_dir)

print(f"Downloading analogy benchmark... ", end="", flush=True)
url = "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt"
response = requests.get(url)
analogies_raw = response.text.split('\n')
analogy_dataset = {}
cur_category = None
for line in analogies_raw:
    if len(line) == 0:
        continue
    if line[0] == ':':
        cur_category = line[1:]
        analogy_dataset[cur_category] = []
        continue
    analogy = line.lower().split(' ')
    analogy_dataset[cur_category].append(analogy)
data_fm.save(analogy_dataset, "analogies.pickle")
print("done.")
