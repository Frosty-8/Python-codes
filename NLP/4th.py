import spacy

nlp = spacy.load('en_core_web_sm')
sentence = 'We commit ourselves to provide quality education'
doc = nlp(sentence)

for token in doc:
    print(f'{token.text} -> {token.pos_} ({token.tag_})')