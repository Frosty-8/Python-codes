from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

k_values = range(1, 21)
training_scores = []
test_scores = []

for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train, y_train)
    training_scores.append(knn_k.score(X_train, y_train))
    test_scores.append(knn_k.score(X_test, y_test))
    
plt.figure(figsize=(12, 6))
sns.lineplot(x=k_values, y=training_scores, marker='o', label='Training Score', color='blue')
sns.lineplot(x=k_values, y=test_scores, marker='o', label='Test Score', color='red')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Score')
plt.title('KNN Classifier Performance')
plt.legend()
plt.grid()
plt.show()