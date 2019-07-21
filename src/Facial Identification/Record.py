import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recog = cv2.face.LBPHFaceRecognizer_create()
recog.read("TrainData.yml")
cap = cv2.VideoCapture(0)

while(True):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray , scaleFactor=2.5, minNeighbors=5)
    for (x,y,w,h) in faces:
    	roi_gray = gray[y:y+h,x:x+w]
    	roi_color = frame[y:y+h,x:x+w]
    	ID,conf = recog.predict(roi_gray)
    	print(ID,conf)    
    	img_item = "my-image.png"
    	cv2.imwrite(img_item,roi_color)

    	color = (255,0,0) #BGR
    	stroke = 2 #thickness
    	cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)
    cv2.imshow('Cam',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

