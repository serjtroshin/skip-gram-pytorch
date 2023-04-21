# pw2v
This repository contains code for training your own skip-gram model

## Download data and building vocab
```
bash build_vocab.sh
```

## Training
```
python pw2v/trainer.py
```
The model will be tested on a bunch of similarty benchmarks: MEN, SImLex-999, wordsim353.