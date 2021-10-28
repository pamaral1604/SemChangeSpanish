import numpy as np
import gensim, sys, scipy
from convert2hist import write_pickle

"""
from: https://gist.github.com/tangert/106822a0f56f8308db3f1d77be2c7942
modified by: Hai Hu

"""

usage="""
python gensim_word2vec_procrustes_align.py model1 model2

model1 will be fixed. model2 will be changed to be aligned to model1. 
"""

def main():
	if len(sys.argv) < 3:
		print(usage)
		exit()
	m1 = gensim.models.Word2Vec.load(sys.argv[1])
	m2 = gensim.models.Word2Vec.load(sys.argv[2])
	print('#vocab m1:', len(m1.wv.vocab))
	print('#vocab m2:', len(m2.wv.vocab))

	m2_new = smart_procrustes_align_gensim(m1, m2)

	####################################
	# SANITY CHECK: BEGIN
	####################################

	print('#vocab m2_new:', len(m2_new.wv.vocab))
	print('#vocab m1:', len(m1.wv.vocab))
	print('#vocab m2:', len(m2.wv.vocab))

	# print('\nfix the model, distance bt algo and nada:')
	# print('- m1:', scipy.spatial.distance.cosine(m1.wv['algo'], m1.wv['nada']))
	# print('- m2:', scipy.spatial.distance.cosine(m2.wv['algo'], m2.wv['nada']))
	# print('- m2_new:', scipy.spatial.distance.cosine(m2_new.wv['algo'], m2_new.wv['nada']))

	# print('\nfix the word, distance bt m1 and m2_new')
	# print('- algo:', scipy.spatial.distance.cosine(m1.wv['algo'], m2_new.wv['algo']))
	# print('- nada:', scipy.spatial.distance.cosine(m1.wv['nada'], m2_new.wv['nada']))
	
	####################################
	# SANITY CHECK: END
	####################################

	### save to histwords
	# prefix = '/media/hai/U/tools/histwords/embeddings/spa/'
	prefix = './'

	vecs = m1.wv.vectors
	vocab = m1.wv.index2word
	np.save(prefix + '1500-w.npy',  vecs)
	write_pickle(vocab, prefix + '1500-vocab.pkl')
	
	vecs = m2_new.wv.vectors
	vocab = m2_new.wv.index2word
	np.save(prefix + '2000-w.npy',  vecs)
	write_pickle(vocab, prefix + '2000-vocab.pkl')

	### save the vocab
	# with open('vocab-intersect.txt','w') as f:
	# 	for i in range(len(vocab)): f.write(vocab[i] + '\n')
	
	return
	
def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
	"""Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
	Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
		(With help from William. Thank you!)

	First, intersect the vocabularies (see `intersection_align_gensim` documentation).
	Then do the alignment on the other_embed model.
	Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
	Return other_embed.

	If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
	"""
	
	# patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
	base_embed.init_sims()
	other_embed.init_sims()

	# make sure vocabulary and indices are aligned
	in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)

	# get the embedding matrices
	base_vecs = in_base_embed.wv.syn0norm
	other_vecs = in_other_embed.wv.syn0norm

	# just a matrix dot product with numpy
	m = other_vecs.T.dot(base_vecs) 
	# SVD method from numpy
	u, _, v = np.linalg.svd(m)
	# another matrix operation
	ortho = u.dot(v) 
	# Replace original array with modified one
	# i.e. multiplying the embedding matrix (syn0norm)by "ortho"
	other_embed.wv.syn0norm = other_embed.wv.syn0 = (other_embed.wv.syn0norm).dot(ortho)
	return other_embed
	
def intersection_align_gensim(m1,m2, words=None):
	"""
	Intersect two gensim word2vec models, m1 and m2.
	Only the shared vocabulary between them is kept.
	If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
	Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
	These indices correspond to the new syn0 and syn0norm objects in both gensim models:
		-- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
		-- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
	The .vocab dictionary is also updated for each model, preserving the count but updating the index.
	"""

	# Get the vocab for each model
	vocab_m1 = set(m1.wv.vocab.keys())
	vocab_m2 = set(m2.wv.vocab.keys())

	# Find the common vocabulary
	common_vocab = vocab_m1&vocab_m2
	if words: common_vocab&=set(words)

	# If no alignment necessary because vocab is identical...
	if not vocab_m1-common_vocab and not vocab_m2-common_vocab:
		return (m1,m2)

	# Otherwise sort by frequency (summed for both)
	common_vocab = list(common_vocab)
	common_vocab.sort(key=lambda w: m1.wv.vocab[w].count + m2.wv.vocab[w].count,reverse=True)

	print(common_vocab[:500])

	# Then for each model...
	for m in [m1,m2]:
		# Replace old syn0norm array with new one (with common vocab)
		indices = [m.wv.vocab[w].index for w in common_vocab]
		old_arr = m.wv.syn0norm
		new_arr = np.array([old_arr[index] for index in indices])
		m.wv.syn0norm = m.wv.syn0 = new_arr

		# Replace old vocab dictionary with new one (with common vocab)
		# and old index2word with new one
		m.wv.index2word = common_vocab
		old_vocab = m.wv.vocab
		new_vocab = {}
		for new_index, word in enumerate(common_vocab):
			old_vocab_obj=old_vocab[word]
			new_vocab[word] = gensim.models.word2vec.Vocab(index=new_index, count=old_vocab_obj.count)
		m.wv.vocab = new_vocab

	return (m1,m2)

if __name__ == "__main__":
	main()
