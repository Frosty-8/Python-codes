import spacy
nlp = spacy.load('en_core_web_sm')
text = 'Apple Inc. is looking at buying U.K. startup for $1 billion'
doc = nlp(text)
print('Named Entities')
for ent in doc.ents:
    print(f'{ent.text} ({ent.label_})')