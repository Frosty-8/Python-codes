# Import necessary libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib

# Step 1: Load the 20 Newsgroups dataset
print("Loading the 20 Newsgroups dataset...")
newsgroups = fetch_20newsgroups(subset='all')

print(newsgroups)

# Step 2: Split the dataset into training and testing sets
print("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.3, random_state=42)

# Step 3: Convert text data into TF-IDF features
print("Transforming text data into TF-IDF features...")
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 4: Train a Multinomial Naive Bayes classifier
print("Training the Multinomial Naive Bayes classifier...")
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Step 5: Make predictions on the test set
print("Making predictions on the test set...")
y_pred = clf.predict(X_test_tfidf)

# Step 6: Evaluate the model
print("Evaluating the model...")
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=newsgroups.target_names))

# Step 7: Perform hyperparameter tuning using GridSearchCV
print("Performing hyperparameter tuning...")
parameters = {'alpha': [0.01, 0.1, 1, 10]}
grid_search = GridSearchCV(MultinomialNB(), parameters, cv=5)
grid_search.fit(X_train_tfidf, y_train)
print("Best parameters:", grid_search.best_params_)

# Step 8: Save the trained model to a file
print("Saving the trained model...")
joblib.dump(clf, 'text_classification_model.pkl')

# Step 9: Load the saved model
print("Loading the saved model...")
clf = joblib.load('text_classification_model.pkl')
print("Model loaded successfully!")