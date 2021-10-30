import helpers
import sys


"""
Let's examine the closest neighbors for a word over time
"""

import numpy as np
import matplotlib.pyplot as plt


# We accept a list of words from command line
# to generate graphs for.

WORDS = helpers.get_words() # words from cmd


if __name__ == "__main__":
    embeddings = helpers.load_embeddings('embeddings/spa') ### Spanish
    # embeddings = helpers.load_embeddings('embeddings/sgns') ### English
    for word1 in WORDS:
        helpers.clear_figure()
        time_sims, lookups, nearests, sims = helpers.get_time_sims(embeddings, word1)

        # print('\n\ntime_sims')
        # print(time_sims)
        # print('\n\nlookups')
        # print(lookups)
        # print('\n\nnearests')
        # print(nearests)
        # # save nearests
        # with open('nearests.txt','w') as f:
        #     for k in nearests.keys():
        #         f.writelines([str(i)+'\n' for i in nearests[k]])
        #         print('\n\n')

        # print('\n\nsims')
        # print(sims)
        # continue

        words = lookups.keys() # neighbors in all time periods ['w1|1890', 'w2|1900' ...]
        # print('\nwords:', words)
        values = [ lookups[word] for word in words ]
        print('\nlen(values):', len(values))
        fitted = helpers.fit_tsne(values)
        if not len(fitted):
            print "Couldn't model word", word1
            continue

        # draw the words onto the graph
        cmap = helpers.get_cmap(len(time_sims))
        annotations = helpers.plot_words(word1, words, fitted, cmap, sims)

        if annotations:
            helpers.plot_annotations(annotations)

        helpers.savefig("%s_annotated" % word1)
        sims = []
        for year, sim in time_sims.iteritems():
            print year, sim
            #sims.append( [sim[1]].split('|')[0] )

