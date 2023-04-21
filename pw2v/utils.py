import math
import torch
import numpy as np

from scipy.stats import spearmanr
from tqdm import tqdm
from multiprocessing import Pool
import csv



def probability_of_keeping_word(freq: float):
    # freq is the frequency of the word in the corpus from 0 to 1
    # return the probability of keeping the word
    # p = (math.sqrt(freq/0.001)+1)*0.001/freq
    p = (math.sqrt(freq / 0.001) + 1) * 0.001 / freq
    return p


def print_nearest(vocab, embeddings, word, k=10):
    # normalize embeddings
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    word_id = word2idx[word]
    word_embedding = embeddings[word_id]
    # normalize
    similarities = torch.matmul(embeddings, word_embedding)
    topv, topk = torch.topk(similarities, k)
    for idx, v in zip(topk, topv):
        print(vocab[idx.item()], v.item())

test_file = {
        "wordsim353" : './data/wordsim353/combined.csv',
        "men" : './data/MEN/MEN_dataset_natural_form_full',
        "simlex" : './data/SimLex-999/SimLex-999.txt'
    }

def get_emb(vec_file):
    f = open(vec_file, 'r', errors='ignore')
    contents = f.readlines()[1:]
    word_emb = {}
    vocabulary = {}
    vocabulary_inv = {}
    for i, content in enumerate(contents):
        content = content.strip()
        tokens = content.split()
        word = tokens[0]
        vec = tokens[1:]
        try:
            vec = np.array([float(ele) for ele in vec])
        except ValueError:
            continue
        word_emb[word] = np.array(vec)
        vocabulary[word] = i
        vocabulary_inv[i] = word

    return word_emb, vocabulary, vocabulary_inv


def read_sim_test(test="wordsim353"):
    f = open(test_file[test])
    if test == 'wordsim353':
        csv_reader = csv.reader(f, delimiter=',')
        tests = {}
        for i, row in enumerate(csv_reader):
            if i > 0:
                word_pair = (row[0].lower(), row[1].lower())
                tests[word_pair] = float(row[2])
    elif test == 'men':
        tests = {}
        for line in f:
            tmp = line.split(" ")
            if len(tmp) != 3:
                continue
            word_pair = (tmp[0].lower(), tmp[1].lower())
            tests[word_pair] = float(tmp[2])
    elif test == 'simlex':
        tests = {}
        for i, line in enumerate(f):
            if i == 0:
                continue
            tmp = line.split("\t")
            if len(tmp) != 10:
                continue
            word_pair = (tmp[0].lower(), tmp[1].lower())
            tests[word_pair] = float(tmp[3])
    return tests


def calc_sim(w1, w2):
    return np.dot(w1, w2)/np.linalg.norm(w1)/np.linalg.norm(w2)

def test_sim(word_emb, tests):
    pool = Pool(10)
    real_tests = {}
    for word_pair in tests:
        w1 = word_pair[0]
        w2 = word_pair[1]
        if w1 in word_emb and w2 in word_emb:
            real_tests[word_pair] = tests[word_pair]
    print(f'{len(real_tests)}/{len(tests)} actual test cases!')
    args = [(word_emb[word_pair[0]], word_emb[word_pair[1]]) for word_pair in real_tests.keys()]
    res = pool.starmap(calc_sim, args)
    truth = list(real_tests.values())
    rho = spearmanr(truth, res)[0]
    print(f'Spearman coefficient: {rho}')
    return rho


def test(path) -> dict:
    word_emb, vocabulary, vocabulary_inv = get_emb(path)
    metrics = {}
    for test in test_file:
        print(f'Evaluation on {test}...')
        tests = read_sim_test(test)
        metrics[test] = test_sim(word_emb, tests)
    return metrics