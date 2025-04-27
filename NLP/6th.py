import spacy
from sklearn.metrics import precision_score, recall_score, f1_score
nlp = spacy.load("en_core_web_sm")

def get_metrics(pred, true):
    y_true = [ent in true for ent in pred]
    y_pred = [True] * len(pred)
    return precision_score(y_true, y_pred, zero_division=0), \
           recall_score(y_true, y_pred, zero_division=0), \
           f1_score(y_true, y_pred, zero_division=0)

text = input("Enter sentence: ")
true = input("True entities (comma-separated): ").split(",")

true = [e.strip() for e in true]
pred = [ent.text for ent in nlp(text).ents]

print("\nNamed Entities:", pred)
p, r, f = get_metrics(pred, true)
print(f"\nPrecision: {p:.2f}  Recall: {r:.2f}  F1: {f:.2f}")