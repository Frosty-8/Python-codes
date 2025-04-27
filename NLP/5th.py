import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

nltk.download('punkt_tab')

text = ' We commit ourselves to provide quality education '
tokens = word_tokenize(text)

unigram_list = list(ngrams(tokens,1)) 
bigram_list = list(ngrams(tokens,2))
trigram_list = list(ngrams(tokens,3))

unigram_freq = FreqDist(unigram_list)
bigram_freq = FreqDist(bigram_list)
trigram_freq = FreqDist(trigram_list)

print("Unigram ")
for unigram,frequency in unigram_freq.items():
    print(f"{unigram} : {frequency}")

print("\nBigram ")
for bigram,frequency in bigram_freq.items():
    print(f"{bigram} : {frequency}")

print("\nTrigram ")
for trigram,frequency in trigram_freq.items():
    print(f"{trigram} : {frequency}")