from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import pickle
import cv2
import numpy as np

print("Load Model")
model = load_model("model.h5")
lb = pickle.loads(open("labelku.lb", "rb").read())

image = cv2.imread("testAnjing.jpg")
tampil = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224)).astype("float32")

image = img_to_array(image)
image = np.expand_dims(image, axis=0)

proba = model.predict(image)[0]
print(proba)
idx = np.argmax(proba)
hasil = lb.classes_[idx]
print(hasil)

cv2.putText(tampil, hasil, (40, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
cv2.imshow("Hasil", tampil)
cv2.waitKey(0)
