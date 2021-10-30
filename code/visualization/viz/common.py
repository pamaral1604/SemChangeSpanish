import sys
import os

import time


tpath = os.path.dirname(os.path.realpath(__file__))
VIZ_DIR=os.path.join(tpath, "web")

ROOT_DIR=tpath
tpath = os.path.abspath(os.path.join(tpath, "../"))

sys.path.append(tpath)
os.chdir(tpath)

import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.manifold import TSNE

from representations.sequentialembedding import SequentialEmbedding

DEGREE_ADVS = ['muy', 'mucho', 'tan', 'poco']

def get_words():
    WORDS = [ "car" ]
    if len(sys.argv) > 1:
        WORDS = sys.argv[1:]

    return WORDS

CMAP_MIN=5
def get_cmap(n, name='YlGn'):
    return plt.cm.get_cmap(name, n+CMAP_MIN)

# this is based on embedding.py get_time_sims
def get_time_sims(self, word1):
    start = time.time()
    time_sims = collections.OrderedDict()  # {'1990': [(0.8343, "car|1990"), ...], '1991': [] }
    lookups = {}   # {"car|1990": vector, ... }
    nearests = {}  # {'car|1950': [(1.00, u'car|1950'), (0.62, u'truck|1950') ...], 'car|1970': [...], ... }
    sims = {}      # {u'pony|1860': 0.3333837489139372, u'railway|1910': 0.36802214765314756, ...}
    for year, embed in self.embeds.iteritems():
        print '\n', year
        nearest = []
        nearests["%s|%s" % (word1, year)]= nearest  # nearests["awful|1900"] = []
        time_sims[year] = []  # time_sims[1900] = []

        # for sim, word in embed.closest(word1, n=15): #original
        lst = embed.closest(word1, n=40)
        neighbors = [i[1] for i in lst]

        # if word1 == 'algo':
        #     for w in DEGREE_ADVS:
        #         if w not in neighbors:
        #             print 'adding', w, embed.similarity(word1, w)
        #             lst.append((embed.similarity(word1, w), w))

        for tup in lst:  # hai
            sim, word = tup[0], tup[1]
            ww = "%s|%s" % (word, year)  # "car|1990"
            nearest.append((sim, ww))    # (0.8343, "car|1990")
            # if sim > 0.3: #original
            if sim > 0.05: # hai
                time_sims[year].append((sim, ww))     # (0.8343, "car|1990")
                lookups[ww] = embed.represent(word)   # lookups["car|1990"] = vector for car in 1990
                # print('\nww, lookups[ww]')
                # print(ww,lookups[ww].shape)
                sims[ww] = sim

    print "GET TIME SIMS FOR %s TOOK %s" % (word1, time.time() - start)
    return time_sims, lookups, nearests, sims

EMBED_CACHE = {}

def clear_embed_cache():
    global EMBED_CACHE
    EMBED_CACHE = {}

import threading
embed_lock = threading.Lock()

EMBEDDING="embeddings/eng-all_sgns"
def load_embeddings(filename=None):
    if not filename:
        filename = EMBEDDING

    with embed_lock:
        print "LOADING EMBEDDINGS %s" % filename
        start = time.time()

        if filename in EMBED_CACHE:
            return EMBED_CACHE[filename]

        print "THIS MIGHT TAKE A WHILE..."

        if filename == 'embeddings/spa':
            embeddings = SequentialEmbedding.load(filename, range(1500, 2500, 500))
        else:
            embeddings = SequentialEmbedding.load(filename, range(1840, 2000, 10))
        print "LOAD EMBEDDINGS TOOK %s" % (time.time() - start)

        EMBED_CACHE[filename] = embeddings
        return embeddings

def get_embedding_list(dirname="embeddings"):
    import stat
    dirs = []
    for f in os.listdir(dirname):
        fname = os.path.join(dirname, f)

        if os.path.isdir(fname):
            dirs.append(fname)

    return dirs

def select_embedding():
    global EMBEDDING
    print ""
    print "Please select an embedding to load"
    embeddings = get_embedding_list()
    for i, embed in enumerate(embeddings):
        print "%s) %s" % (i+1, embed)

    while True:
        selection = raw_input("Load which embedding? ")
        try:
            select_num = int(selection)
            embedding = embeddings[select_num-1]
            break
        except:
            print "Please select a number between %s and %s" % (1, len(embeddings))

    print ""
    EMBEDDING = embedding

    return load_embeddings(embedding)


def clear_figure():
    plt.figure(figsize=(20,20))
    plt.clf()

def fit_tsne(values):
    if not values:
        return

    start = time.time()
    mat = np.array(values)
    model = TSNE(n_components=2, random_state=0, learning_rate=150, init='pca', n_iter=3000, perplexity=15)
    fitted = model.fit_transform(mat)
    print "FIT TSNE TOOK %s" % (time.time() - start)

    return fitted


def get_now():
    return int(time.time() * 1000)



# plotting features, not used much any more
def plot_words(word1, words, fitted, cmap, sims):
    # TODO: remove this and just set the plot axes directly
    plt.scatter(fitted[:,0], fitted[:,1], alpha=0)
    plt.suptitle("%s" % word1, fontsize=30, y=0.1)
    plt.axis('off')

    annotations = []
    isArray = type(word1) == list
    for i in xrange(len(words)):
        pt = fitted[i]

        ww,decade = [w.strip() for w in words[i].split("|")]
        # for English
        if int(decade) not in [1500, 2000]:
            color = cmap((int(decade) - 1840) / 10 + CMAP_MIN)
        # Spanish
        else:
            if int(decade) == 1500: color = (1,0,0,1)
            elif int(decade) == 2000: color = (0,0,.90,1)
        # print('**********')
        # print(color)
        word = ww
        sizing = 25
        # sizing = sims[words[i]] * 30

        # word1 is the word we are plotting against
        if ww == word1 or (isArray and ww in word1):
            # old: plot the year
            # annotations.append((ww, decade, pt))
            # word = decade  

            # Hai: plot the word_decade
            annotations.append((ww, decade, pt))
            word = ww + '_' + str(decade)
            color = 'black'
            sizing = 30

        bbox=None
        # Hai: add round box for degree advs
        # if word in DEGREE_ADVS + ['algo']:
        #     bbox=dict(boxstyle="round,pad=0.3", fc="w", 
        #               color='black', alpha=0.6)

        # Hai: add round box for word1
        if '_' in word: 
            bbox=dict(boxstyle="round,pad=0.3", fc="w", 
                color='black', alpha=0.6)

        plt.text(pt[0], pt[1], word, color=color, size=int(sizing), bbox=bbox)

    return annotations

def plot_annotations(annotations):
    # draw the movement between the word through the decades as a series of
    # annotations on the graph
    annotations.sort(key=lambda w: w[1], reverse=True)
    prev = annotations[0][-1]
    for ww, decade, ann in annotations[1:]:
        plt.annotate('', xy=prev, xytext=ann,
            arrowprops=dict(facecolor='blue', shrink=0.1, alpha=0.3,width=2, headwidth=15))
        print prev, ann
        prev = ann

def savefig(name):

    directory = os.path.join(ROOT_DIR, "output")
    if not os.path.exists(directory):
        os.makedirs(directory)

    fname = os.path.join(directory, name)

    plt.savefig(fname, bbox_inches=0, dpi=300)

