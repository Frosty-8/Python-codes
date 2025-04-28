import nltk
from nltk.corpus import wordnet as wn
try:
    wn.synsets('dog')
except:
    nltk.download('wordnet')

dog_synset = wn.synset('dog.n.01')
print(f"Word: dog (Synset: {dog_synset.name()})")
print("\nHypernyms:")
for hypernym in dog_synset.hypernyms():
    print(f"- {hypernym.name()}")
print("\nHyponyms (examples):")
for hyponym in dog_synset.hyponyms()[:5]: 
     print(f"- {hyponym.name()}")
