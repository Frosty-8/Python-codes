# import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn import metrics
# import matplotlib.pyplot as plt

# iris = load_iris()
# X = iris.data[iris.target != 0]  
# y = iris.target[iris.target != 0]

# X = X[:, :2]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# svm_classifier = SVC(kernel='linear', random_state=42)
# svm_classifier.fit(X_train, y_train)

# y_pred = svm_classifier.predict(X_test)

# accuracy = metrics.accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# def plot_decision_boundary(X, y, model):
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
#                          np.arange(y_min, y_max, 0.01))
    
#     Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
    
#     plt.contourf(xx, yy, Z, alpha=0.8)
#     plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.title('SVM Decision Boundary')
#     plt.show()

# plot_decision_boundary(X_train, y_train, svm_classifier)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd

# Load the mushroom dataset from the CSV file
mushroom = pd.read_csv("C:/Users/Sarthak/OneDrive/Documents/Desktop/python/ml/mushrooms.csv")
mushroom_df = pd.DataFrame(mushroom)
print(mushroom_df)

# Label encoding for categorical columns
label_encoders = {}
for column in mushroom_df.columns:
    le = LabelEncoder()
    mushroom_df[column] = le.fit_transform(mushroom_df[column])
    label_encoders[column] = le

X = mushroom_df[['cap-shape', 'cap-color']].values
y = mushroom_df['class'].values  # Class represents edible (e) or poisonous (p)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the SVM classifier (linear kernel for decision boundary visualization)
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)

# Predict on the test set and calculate accuracy
y_pred = svm_classifier.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy : ", accuracy)

# Function to plot the decision boundary
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Cap Shape')
    plt.ylabel('Cap Color')
    plt.title('SVM Decision Boundary on Mushroom Data')
    plt.show()

# Plot the decision boundary
plot_decision_boundary(X_train, y_train, svm_classifier)