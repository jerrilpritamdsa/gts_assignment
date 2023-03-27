import cv2
import os
import numpy as np

recognizer = cv2.face_LBPHFaceRecognizer.create()
recognizer.read('gts_assignment/trained_model.xml')

path = 'gts_assignment/Sample_For_Python_Scripting'

output_path = 'gts_assignment/test_folder'

label_dict = {
    1: 'person_1',
    2: 'person_2',
    3: 'person_3',
    4: 'person_4',
    5: 'person_5',
    6: 'person_6',
    7: 'person_7',
    8: 'person_8',
    9: 'person_9',
    10: 'person_10',
    11: 'person_11',
    12: 'person_12',
    13: 'person_13',
    14: 'person_14',
    15: 'person_15'
}

for i in range(1, 16):
    folder_path = os.path.join(output_path, f'person_{i}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


for filename in os.listdir(path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        label, confidence = recognizer.predict(img)

        dst_folder = os.path.join(output_path, label_dict[label])
        dst_path = os.path.join(dst_folder, filename)
        os.rename(img_path, dst_path)