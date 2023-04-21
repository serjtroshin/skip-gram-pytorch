from pw2v.vocab import Vocab, UNK
from pathlib import Path
from collections import Counter
import numpy as np

def test_vocab_size():
    vocab = Vocab(str(Path(__file__).parent.joinpath("sample.txt")), min_freq=1)
    assert len(vocab) == 5  # 4 words + UNK

def test_vocab_sample():
    N = 100000
    vocab = Vocab(str(Path(__file__).parent.joinpath("sample.txt")), min_freq=1)
    sample_ids = vocab.sample(N)
    counter = Counter(sample_ids)
    freqs = np.array([1, 2, 3, 4]) ** 0.5
    freqs = freqs / freqs.sum()
    tol = 0.05
    assert abs(counter[vocab.word2idx["one"]] / N - freqs[0]) < tol
    assert abs(counter[vocab.word2idx["two"]] / N - freqs[1]) < tol
    assert abs(counter[vocab.word2idx["three"]] / N - freqs[2]) < tol
    assert abs(counter[vocab.word2idx["four"]] / N - freqs[3]) < tol
    # print([vocab.idx2word[id] for id in sample_ids])

def test_skip_word():
    N = 100000
    vocab = Vocab(str(Path(__file__).parent.joinpath("sample.txt")), min_freq=1)
    sample_ids = vocab.sample(N, skip_word_id=vocab.word2idx["one"])
    assert not vocab.word2idx["one"] in sample_ids

    freqs = np.array([0, 2, 3, 4]) ** 0.5
    freqs = freqs / freqs.sum()
    tol = 0.05
    counter = Counter(sample_ids)
    assert abs(counter[vocab.word2idx["one"]] / N - freqs[0]) < tol
    assert abs(counter[vocab.word2idx["two"]] / N - freqs[1]) < tol
    assert abs(counter[vocab.word2idx["three"]] / N - freqs[2]) < tol
    assert abs(counter[vocab.word2idx["four"]] / N - freqs[3]) < tol

def test_unk():
    vocab = Vocab(str(Path(__file__).parent.joinpath("sample.txt")), min_freq=1)
    assert vocab.word2idx["five"] == vocab.word2idx[UNK]
