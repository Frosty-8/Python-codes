from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.kernel_approximation import RBFSampler
from sklearn import metrics

data = load_iris()
X = data.data 
y = data.target  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% train, 30% test

# 4. Create Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scale features to zero mean and unit variance
    ('rbf_features', RBFSampler(gamma=1, n_components=100, random_state=42)),  # Apply RBF kernel mapping
    ('svm', SVC(kernel='linear'))  # Use linear SVM classifier
])

pipeline.fit(X_train, y_train) 

y_pred = pipeline.predict(X_test)  
print("Accuracy on test set:", metrics.accuracy_score(y_test, y_pred))  
print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred))  