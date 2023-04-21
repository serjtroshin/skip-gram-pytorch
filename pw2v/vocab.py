from __future__ import annotations
import numpy as np
from collections import defaultdict

from utils import probability_of_keeping_word

UNK = "<UNK>"

class Vocab:
    """
    reads file and builds a vocabulary
    """
    def __init__(self, file, min_freq=5):
        self.word2idx = defaultdict(self.create_unk_defaultdict)
        self.idx2word = []
        self.freqs = {}
        self.min_freq = min_freq
        self.file = file
        if self.file is not None:
            self._build_vocab()
            self.prepare_frequences()

    def create_unk_defaultdict(self):
        return self.word2idx[UNK]
    
    @property
    def unk_id(self):
        return self.word2idx[UNK]

    def _build_vocab(self):
        """
        builds vocabulary from file
        """
        with open(self.file, 'r') as f:
            for line in f:
                for word in line.split():
                    self.freqs[word] = self.freqs.get(word, 0) + 1
        for word, freq in sorted(self.freqs.items(), key=lambda x: x[1], reverse=True):
            # sort ids by frequency (decreasing)
            if freq >= self.min_freq:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word.append(word)
        # add UNK token
        assert not UNK in self.word2idx
        self.word2idx[UNK] = len(self.word2idx)
        self.idx2word.append(UNK)
        self.freqs[UNK] = 0

    def save(self, file):
        """
        saves vocabulary to file
        """
        with open(file, 'w') as f:
            for id in range(len(self.idx2word)):
                word = self.idx2word[id]
                f.write(f"{word} {self.freqs[word]} {self.word2idx[word]}\n")

    @staticmethod
    def from_file(file) -> Vocab:
        """
        loads vocabulary from file
        """
        self = Vocab(None)
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                word, freq, idx = line.split()
                idx = int(idx)
                self.word2idx[word] = int(idx)
                assert i == idx, (i, idx)
                self.idx2word.append(word)
                self.freqs[word] = int(freq)
        self.prepare_frequences()
        return self

    def __len__(self):
        return len(self.idx2word)
    
    def prepare_frequences(self):
        """
        squeare root of frequencies and normalization for sampling
        """
        freqs = np.array(list(self.freqs[self.idx2word[idx]] for idx in range(len(self))))
        denom = freqs.sum()
        self.normalized_freqs = freqs / denom

    def to_keep(self, word_id):
        # return True if the word should be kept
        return np.random.rand() < probability_of_keeping_word(self.normalized_freqs[word_id])
    
    # def sample(self, size, skip_word_id=None) -> np.ndarray:
    #     """
    #     samples from a vocabulary proportional to squared root of a frequency
    #     if skip_word_id is not None, then it will not be sampled
    #     return word ids as np array
    #     """
    #     denom = None
    #     if skip_word_id is not None:
    #         # set freq to 0
    #         self._normalized_freqs_tmp[:] = self.normalized_freqs
    #         self.normalized_freqs[skip_word_id] = 0
    #         self.normalized_freqs /= self.normalized_freqs.sum()

    #     ids = np.random.choice(len(self.normalized_freqs), size=size, p=self.normalized_freqs, replace=True)


    #     if skip_word_id is not None:
    #         # return freqs to normal
    #         self.normalized_freqs[:] = self._normalized_freqs_tmp  # return back to original freqs


    #     return ids



if __name__=="__main__":
    import logging

    logger = logging.getLogger("vocab.py")
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/text8")
    parser.add_argument("--min_freq", type=int, default=5)
    parser.add_argument("--save", type=str, default="data/vocab.txt")
    args = parser.parse_args()

    logger.info(f"Building vocabulary from {args.file}")
    logger.info(f"Minimum frequency: {args.min_freq}")
    vocab = Vocab(args.file, args.min_freq)
    vocab.save(args.save)
    logger.info(f"Vocabulary size: {len(vocab)}")