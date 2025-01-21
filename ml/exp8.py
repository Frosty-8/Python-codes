from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score as SS
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
    
iris = load_iris()
X=iris.data

kmeans = KMeans(n_clusters=3,random_state=42)
kmeans.fit(X)

centers = kmeans.cluster_centers_
labels=kmeans.labels_

s_avg=SS(X,labels)
print(f"Silhoutte Score : {s_avg:2f}")

plt.scatter(X[:,0],X[:,1],c=labels,marker='o')
plt.scatter(centers[:,0],centers[:,1],c='red',marker='x',s=100)
plt.show()