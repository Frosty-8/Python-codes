from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np
import joblib

def main():
    print("Loading 20 newsgroups dataset for categories:")
    print("Options : ")
    print("1. Train and Evaluate Model  ")
    print("2. Load Existing Model  ")
    print("3. Enter a statement for analysis  ")
    choice = input("Enter your choice(1/2/3) = ")

    if choice == '1':
        train_and_evaluate()
    elif choice == '2':
        evaluate_loaded_model()
    elif choice == '3':
        analyze_statement() 
    else:
        print("Invalid Choice")

def train_and_evaluate():
    print("Fetching trraining dataset.....")
    twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    print("Training the model.....")
    text_clf.fit(twenty_train.data, twenty_train.target)

    model_filename=input("Enter filename to save the model (e.g., 'model.pkl') : ")
    joblib.dump(text_clf, model_filename)
    print("Model saved as ", model_filename)

    evaluate_model(text_clf, "train")

def evaluate_model(model,subset="test"):
    print("Fetching {subset} dataset.....")
    data = fetch_20newsgroups(subset=subset, shuffle=True)

    print("Making predictions....")
    predicted = model.predict(data.data)
    
    accuracy = np.mean(predicted == data.target)
    print(f"Accuracy : {accuracy:.4f}")

    print("Classification Report : ")
    print(metrics.classification_report(data.target, predicted, target_names=data.target_names))

def evaluate_loaded_model():
    model_filename = input("Enter filename to load the model (e.g., 'model.pkl') : ")
    try:
        loaded_model = joblib.load(model_filename)
        print(f"Model loaded from {model_filename}")

        evaluate_model(loaded_model,"test")
    except FileNotFoundError:
        print("File not found")

def analyze_statement():
    model_filename = input("Enter filename to load the model (e.g., 'model.pkl') : ")
    try:
        loaded_model = joblib.load(model_filename)
        print(f"Model loaded from {model_filename}")

        statement = input("Enter statement to analyze : ")
        prediction = loaded_model.predict([statement])
        target_names = fetch_20newsgroups(subset='train').target_names
        print(f"The statement is classified as {target_names[prediction[0]]}")
    except FileNotFoundError:
        print("File not found")

if __name__ == "__main__":
    main()