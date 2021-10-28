"""
convert gensim model to `histword` format, which has 2 files:
- mat = ...-w.npy : a numpy mat
- vocab = ...-vocab.pkl : a list of words, descreasing freq

"""

import numpy as np
import gensim
import pickle

def main():
    pass

def test():
    ### test how to save as histwords format

    # chr
    # old: spellings have not been normalized
    fn = 'gensim/corpus-chronicles_lower-1_prepro-gensim_dim-100_wsize-7_mincnt-20_sg-1_neg-5_seed-1.model'
    # new: normalized spelling
    fn = 'gensim/corpus-chronicles.n_lower-1_prepro-gensim_dim-100_wsize-7_mincnt-20_sg-1_neg-5_expo-1_seed-1.model'
    _save(fn, '/media/hai/U/tools/histwords/embeddings/spa/1500')
    _load('/media/hai/U/tools/histwords/embeddings/spa/1500')

    # SBW
    #fn = 'SBW_gensim/corpus-modern_lower-1_prepro-gensim_dim-100_wsize-7_mincnt-20_sg-1_neg-5_seed-1.model'
    #_save(fn, '/media/hai/U/tools/histwords/embeddings/spa/2000')
    #_load('/media/hai/U/tools/histwords/embeddings/spa/2000')

def _save(fn, output_prefix):
    m = gensim.models.Word2Vec.load(fn)
    vecs = m.wv.vectors
    vocab = m.wv.index2word
    np.save(output_prefix + '-w.npy',  vecs)
    write_pickle(vocab, output_prefix + '-vocab.pkl')

def _load(fn_prefix):
    vecs = np.load(fn_prefix + '-w.npy', mmap_mode="c")
    vocab = load_pickle(fn_prefix + '-vocab.pkl')
    print(type(vecs))
    print(vecs.shape)
    print(len(vocab))

def write_pickle(data, filename):
    fp = open(filename, "wb")
    pickle.dump(data, fp, protocol=2)

def load_pickle(filename):
    fp = open(filename, "rb")
    return pickle.load(fp)

if __name__ == "__main__":
    main()

