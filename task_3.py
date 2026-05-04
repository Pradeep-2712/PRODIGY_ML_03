import os
print("Current working directory:", os.getcwd())
print("Files here:", os.listdir())

import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf

base_model = VGG16(weights='imagenet', include_top=False)

dataset_path = "PetImages"
categories = ["Cat", "Dog"]

features = []
labels = []

IMG_SIZE = 224

for category in categories:
    path = os.path.join(dataset_path, category)
    label = categories.index(category)

    for img_name in tqdm(os.listdir(path)[:2000]):
        try:
            img_path = os.path.join(path, img_name)

            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)

            feature = base_model.predict(img, verbose=0)[0]
            feature = np.mean(feature, axis=(0, 1))

            features.append(feature)
            labels.append(label)

        except:
            continue

X = np.array(features)
y = np.array(labels)

print("Feature shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))