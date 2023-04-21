""" script for loading embeddings and printing nearest neighbors """
import torch

from pw2v.utils import print_nearest

def load_embeddings(path):
    vocab = []
    embeddings = []
    with open(path, 'r') as f:
        for line in f:
            word, *embedding = line.split()
            if len(embedding) < 10:
                continue
            vocab.append(word)
            embeddings.append(torch.tensor(list(map(float, embedding))))
    return vocab, torch.stack(embeddings)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings", type=str, default="data/fasttext_0.txt")
    parser.add_argument("--word", type=str, default="king")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    vocab, embeddings = load_embeddings(args.embeddings)
    print_nearest(vocab, embeddings, args.word, args.k)