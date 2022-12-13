import tensorflow as tf
from tensorflow.keras import layers, models
from imutils import paths
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle

# create constanta
LABELS = set(["awan", "golden", "husky", "monitor", "pitbull"])

# load image in path
print("Load Gambar .......")
datasetpath = "dataset"

# print(imagePaths)
imagePaths = list(paths.list_images(datasetpath))

# create matrix 3 dimension
data = []
labels = []

# load data
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]

    if label not in LABELS:
        continue

    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    data.append(image)
    labels.append(label)

data = np.array(data)
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.2,
                                                  random_state=20)

setepoch = 20
# ------------
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
# ------------
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(trainX, trainY, epochs=setepoch, batch_size=100,
                    validation_data=(testX, testY))

plt.plot(history.history['accuracy'], label='akurasi')
plt.plot(history.history['val_accuracy'], label='val_akurasi')
plt.xlabel("Epoch")
plt.ylabel("Akurasi")
plt.legend(loc='lower right')

plt.show()

print("INFO Simpan Model .......")
model.save("model.h5", save_format="h5")

print("INFO Simpan Label .......")
f = open("labelku.lb", "wb")
f.write(pickle.dumps(lb))
f.close()
