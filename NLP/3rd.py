import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

text = 'Apple Inc. is looking at buying U.K. startup for $1 billion'
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)

print("\nPart-of-Speech Tags:")
for word, tag in pos_tags:
    print(f"{word}: {tag}")