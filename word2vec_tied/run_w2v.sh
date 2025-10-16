make
DATA_FILE="$DATASETPATH/qwem/cocanow/cocanow.txt"
if [ ! -e "$DATA_FILE" ]; then
  echo "Error: file not found at $DATA_FILE"
  exit 1
fi
time ./word2vec -train "$DATA_FILE" -output vectors.bin -size 200 -min-count 20 -window 16 -negative 1 -sample 1e-5 -threads 20 -binary 1 -iter 10
./compute-accuracy vectors.bin 30000 < questions-words.txt
