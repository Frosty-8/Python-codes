import gensim.downloader as api
from rich import print as rprint

rprint(list(api.info()['models'].keys()))

glove_model = api.load("glove-wiki-gigaword-50")

rprint("Vector for 'computer':", glove_model['computer'])
rprint("Similarity between 'computer' and 'laptop': ", glove_model.similarity('computer','laptop'))