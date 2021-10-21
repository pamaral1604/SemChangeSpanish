"""
compute the analogy score of an embedding model
Hai Hu, Feb 2020
"""

import sys, os, re
import gensim.models
import numpy as np
import tempfile
# from sklearn.metrics.pairwise import cosine_similarity
# from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
# from gensim.scripts.glove2word2vec import glove2word2vec


def main():
    usage='python analogy_score.py fn_analogy_gold fn_model '
    if len(sys.argv) != 3: 
        print(usage)
        exit()
    fn_model = sys.argv[2]
    fn_analogy_gold = sys.argv[1]
    compute(fn_model, fn_analogy_gold)

def compute(fn_model, fn_analogy_gold):
    #print(fn_model)
    #print(fn_analogy_gold)
    # documentation on analogy test:
    # https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.FastTextKeyedVectors.evaluate_word_analogies
    restrict_vocab, dum4unk = 1000000, False
    try:
        model = gensim.models.Word2Vec.load(fn_model)
        print('gensim model')
        analogy_scores = model.wv.evaluate_word_analogies(fn_analogy_gold, restrict_vocab, dummy4unknown=dum4unk) # 0 acc for OOV words
    except: 
        try: # glove model
            wv = KeyedVectors.load_word2vec_format(fn_model, binary=False)
            print('txt format; binary False')
        except: # SBW downloaded; binary = True
            wv = KeyedVectors.load_word2vec_format(fn_model, binary=True)
            print('txt format; binary True')
        analogy_scores = wv.evaluate_word_analogies(fn_analogy_gold, restrict_vocab, dummy4unknown=dum4unk)
    print('section\taccuracy\tn_tests')
    for sec in analogy_scores[1]:
        #print(sec)
        n_correct = len(sec['correct'])
        n_total = len(sec['correct'])+len(sec['incorrect'])
        try: print('{}\t{}\t{}'.format(sec['section'].replace(' ','-'), n_correct/n_total, n_total ))
        except ZeroDivisionError:
            print('{}\t{}\t{}'.format(sec['section'].replace(' ','-'), 'all-OOV', 'all-OOV'))
    #print('total\t{}'.format(analogy_scores[0]))

if __name__ == "__main__":
    main()

