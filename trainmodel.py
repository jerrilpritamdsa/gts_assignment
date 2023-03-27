import cv2
import os
import numpy as np

path = 'gts_assignment/Sample_For_Python_Scripting/test'
labels = []
images = []

for filename in os.listdir(path):
    if filename.endswith('.jpg'):
        img_path = os.path.join(path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = len(labels) + 1
        labels.append(label)
        images.append(img)


recognizer = cv2.face_LBPHFaceRecognizer.create()
recognizer.train(images, np.array(labels, dtype=np.int32))

recognizer.save('gts_assignment/trained_model.xml')