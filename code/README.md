## Code for training word2vec models, alignment of models, and visualization

## files

### analogy

- `analogy_score.py`: compute the score of a trained word2vec model on some analogy test
- `our_analogy_test.txt`: analogy tests for medieval Spanish
- `questions-words_sp.txt`: analogy tests translated from the Google English tests

### training word2vec

- `train_word2vec.py`: use the `gensim` library to train word2vec models
- `chronicles_clean_all.txt.clean`: cleaned chronicles corpus, used for training the models

### alignment of models

- `convert2hist.py`: script containing helper functions that convert word2vec models to formats that are used in `histwords` project (https://nlp.stanford.edu/projects/histwords/)
- `gensim_word2vec_procrustes_align_20201205.py`: to perform alignment

