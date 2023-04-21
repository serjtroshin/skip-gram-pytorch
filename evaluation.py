from pw2v.utils import read_sim_test, get_emb, test_sim, test_file

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vec_file', type=str, default='./data/fasttext_9.txt')

    args = parser.parse_args()
    word_emb, vocabulary, vocabulary_inv = get_emb(args.vec_file)
    for test in test_file:
        print(f'Evaluation on {test}...')
        tests = read_sim_test(test)
        test_sim(word_emb, tests)
