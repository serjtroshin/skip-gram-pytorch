import logging
from vocab import Vocab

from torch.utils.data import Dataset, DataLoader
import numpy as np

import pickle

logger = logging.getLogger("dataloader.py")
logging.basicConfig(level=logging.INFO)

class WordDataset(Dataset):
    def __init__(self, file, vocab_path: str, window_size: int = 5, n_negatives=5, debug=False):
        # check that file chunk is not too large
        self.vocab = self._load_vocab(vocab_path)
        self.file = file
        self.window_size = window_size
        self._build_dataset(debug)
        self.neg_table = self._init_sample_table()
        self.n_negatives = n_negatives

    def _init_sample_table(self):
        freqs = np.array(list(self.vocab.freqs[self.vocab.idx2word[idx]] for idx in range(len(self.vocab))))
        freqs = freqs ** 0.75
        denom = freqs.sum()
        freqs /= denom

        table_size = 1e8
        count = np.round(freqs*table_size)
        sample_table = []
        for idx in range(len(freqs)):
            sample_table += [idx]*int(count[idx])
        return np.array(sample_table)

    def _load_vocab(self, vocab_path):
        return Vocab.from_file(vocab_path)
    
    def to_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def _build_dataset(self, debug):
        # iterate over file and build dataset with word ngrams
        # add current word id and context words ids to dataset
        self.dataset = []
        n_skipped = 0
        with open(self.file, 'r') as f:
            if debug:
                f = [f.read(100000)]
            for line in f:
                words = line.split()
                for i, word in enumerate(words):
                    word_id = self.vocab.word2idx[word]
                    if word_id == self.vocab.unk_id:
                        continue
                    if not self.vocab.to_keep(word_id):
                        n_skipped += 1
                        continue
                    context_ids = []
                    for j in range(i - self.window_size, i + self.window_size + 1):
                        if j == i or j < 0 or j >= len(words):
                            continue
                        context_ids.append(self.vocab.word2idx[words[j]])
                    for context_id in context_ids:
                        if context_id == self.vocab.unk_id:
                            continue
                        if not self.vocab.to_keep(context_id):
                            n_skipped += 1
                            continue
                        self.dataset.append((word_id, context_id))
        self.dataset = np.array(self.dataset)
        logger.info(f"Dataset size: {len(self.dataset)}")
        logger.info(f"Skipped {n_skipped} words (downsampling freq words as in word2vec)")

    def __getitem__(self, idx):
        left, right = self.dataset[idx]
        # sample negatives
        negs = np.random.choice(self.neg_table, size=self.n_negatives)
        return left, right, negs
    
    def __len__(self):
        return len(self.dataset)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/text8")
    parser.add_argument("--vocab", type=str, default="data/text8.vocab")
    parser.add_argument("--window_size", type=int, default=3)
    args = parser.parse_args()
    dataset = WordDataset(args.file, args.vocab, args.window_size, debug=True)
    vocab = dataset.vocab
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=0)
    for batch in dataloader:
        left, right, negs = batch
        print(left.shape, right.shape, negs.shape)
        for i in range(len(left)):
            print(f"({vocab.idx2word[left[i]]}, {vocab.idx2word[right[i]]}), negs: {[vocab.idx2word[neg] for neg in negs[i]]}")
        break