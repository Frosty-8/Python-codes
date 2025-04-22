import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, classification_report, average_precision_score

# Load the dataset
from sklearn.datasets import load_iris

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = (iris.target == 0).astype(int)  # 1 if Setosa, else 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

os.makedirs("results", exist_ok=True)

def save_conf_matrix(model, model_name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'results/confusion_matrix_{model_name}.png')
    plt.close()

save_conf_matrix(log_model, "LogisticRegression")
save_conf_matrix(rf_model, "RandomForest")

def save_pr_curve(model, model_name):
    y_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    avg_prec = average_precision_score(y_test, y_probs)

    plt.plot(recall, precision, label=f"{model_name} (AP={avg_prec:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()

save_pr_curve(log_model, "LogisticRegression")
save_pr_curve(rf_model, "RandomForest")
plt.savefig("results/precision_recall_comparison.png")
plt.close()

print("Logistic Regression Classification Report:")
print(classification_report(y_test, log_model.predict(X_test)))

print("Random Forest Classification Report:")
print(classification_report(y_test, rf_model.predict(X_test)))
