from gensim.models import Word2Vec
from rich import print as rprint

sentences = [
    ['this', 'is', 'the', 'first', 'sentence'],
    ['this', 'is', 'the', 'second', 'sentence'],
    ['and', 'this', 'is', 'the', 'third', 'one']
]
model = Word2Vec(sentences, vector_size=10, window=5, min_count=1, workers=4, sg=1)

rprint('Vector for learning : ', model.wv['this'])
rprint('Most similar words : ', model.wv.most_similar('the'))