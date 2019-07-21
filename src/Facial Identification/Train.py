import cv2
import os
import numpy as np
import pickle
from PIL import Image

recog = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
Base_Dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(Base_Dir,"pics")

y_labels =[]
x_train =[]
label_ids = {}
curr_id =0 
for root,dirs,files in os.walk(img_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ","-").lower()
            if not label in label_ids:
                label_ids[label]=curr_id
                curr_id+=1
                #label_ids[label]= file
            ID = curr_id-1
            pil_image = Image.open(path).convert("L") #grayscale
            img_array = np.array(pil_image,"uint8")
            faces = face_cascade.detectMultiScale(img_array, scaleFactor=2.5, minNeighbors=5)

            for(x,y,w,h) in faces:
                roi = img_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(ID)

with open("labels.pickle","wb") as f:
    pickle.dump(label_ids,f)

recog.train(x_train,np.array(y_labels))
recog.save("TrainData.yml")
    
