from collections import Counter

documents = [
    "Hello world",
    "Hello Hadoop",
    "Hello MapReduce world",
]
words = [word.lower() for doc in documents for word in doc.split()]
word_counts = Counter(words)
for word, count in sorted(word_counts.items()):
    print(f"{word}: {count}")
