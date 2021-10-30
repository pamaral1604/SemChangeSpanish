"""
compute distance between degree adverbs and algo after alignment
Hai Hu
"""

import helpers
import sys

words = ['muy', 'mucho', 'tan', 'poco']

# get two embedding models for Spanish
embeddings = helpers.load_embeddings('embeddings/spa') ### Spanish

for year, embed in embeddings.embeds.iteritems():
    # embed is an Embedding object 
    for word in words: 
        print '\n', word, embed.similarity(word, 'algo')
