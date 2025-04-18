import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def generate_shape(shape, size=32):
    img = np.zeros((size, size), dtype=np.uint8)
    if shape == "circle":
        cv2.circle(img, (16,16), 10, 255, -1)
    elif shape == "square":
        cv2.rectangle(img, (8,8), (24,24), 255, -1)
    elif shape == "triangle":
        pts = np.array([[16,4],[4,28],[28,28]], np.int32)
        cv2.drawContours(img, [pts], 0, 255, -1)
    return img

def create_dataset(n=100):
    X, y, shapes = [], [], ["circle", "square", "triangle"]
    for idx, shape in enumerate(shapes):
        for _ in range(n):
            img = generate_shape(shape)
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            noisy_img = cv2.add(img, noise)
            X.append(noisy_img.flatten())
            y.append(idx)
    return np.array(X), np.array(y), shapes

X, y, labels = create_dataset(n=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=labels))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

for i in range(5):
    img = X_test[i].reshape(32, 32)
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {labels[y_test[i]]} | Pred: {labels[y_pred[i]]}")
    plt.axis('off')
    plt.show()