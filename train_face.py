import os
import cv2
import pickle
import numpy as np
from PIL import Image
def train():
    BASE_DIR= os.path.dirname(os.path.abspath(__file__))
    image_dir=os.path.join(BASE_DIR,"dataset")

    face_cascade=cv2.CascadeClassifier('face-recog\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')
    recognizer=cv2.face.LBPHFaceRecognizer_create()

    current_id=0
    labels_id={}
    y_labels=[]
    x_train=[]
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path=os.path.join(root, file)
                label = os.path.basename(root).lower()
                if not label in labels_id:
                    labels_id[label]=current_id
                    current_id+=1

                id_=labels_id[label]
                pil_image=Image.open(path).convert("L") #grayscale
                image_array=np.array(pil_image, "uint8")
                face=face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
                
                for x,y,w,h in face:
                    roi=image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)

    with open("label.pickle","wb") as f:
        pickle.dump(labels_id,f)
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainner.yml")