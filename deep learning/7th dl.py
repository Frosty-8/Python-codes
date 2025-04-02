import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

selected_classes = [0, 1, 5]  
train_mask = np.isin(y_train, selected_classes)
test_mask = np.isin(y_test, selected_classes)

X_train, y_train = X_train[train_mask], y_train[train_mask]
X_test, y_test = X_test[test_mask], y_test[test_mask]

X_train = np.stack([X_train]*3, axis=-1)
X_test = np.stack([X_test]*3, axis=-1)

X_train = tf.image.resize(X_train, (224, 224)) / 255.0
X_test = tf.image.resize(X_test, (224, 224)) / 255.0

label_map = {val: i for i, val in enumerate(selected_classes)}
y_train = to_categorical([label_map[label] for label in y_train], num_classes=3)
y_test = to_categorical([label_map[label] for label in y_test], num_classes=3)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
out_layer = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=out_layer)
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=32)

for layer in base_model.layers[-5:]:
    layer.trainable = True
model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
history_fine = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=32)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

model.save("fine_tuned_fashion_mnist.h5")