from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

clf_tuned = DecisionTreeClassifier(criterion="entropy", max_depth=3, min_samples_split=4)
clf_tuned.fit(X_train, y_train)

y_pred_tuned = clf_tuned.predict(X_test)

print("\nTuned Accuracy:", metrics.accuracy_score(y_test, y_pred_tuned))

tree=DecisionTreeClassifier()
tree.fit(X,y)
plt.figure(figsize=(12, 8))
plot_tree(tree, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title('Decision Tree for Iris Dataset')
plt.show()