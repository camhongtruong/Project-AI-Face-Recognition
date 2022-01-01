#import các packet cần thiết
import cv2
import numpy as np
import pickle
import os
def recognize():
    face_cascade=cv2.CascadeClassifier('face-recog\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')
    recognizer=cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainner.yml")

    labels={"person_name":1}
    with open("label.pickle","rb") as f:
        orig_labels = pickle.load(f)
        labels={v:k for k,v in orig_labels.items()}

    cap=cv2.VideoCapture(0)
    while (True):
        #bắt từng khung hình
        ret, frame =cap.read()

        #detect faces
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        
        #recognize
        for (x,y,w,h) in faces:
            roi_gray=gray[y:y+h,x:x+w]
            id_, conf=recognizer.predict(roi_gray)
            if conf>=4 and conf<=100:
                font=cv2.FONT_HERSHEY_SIMPLEX
                name=labels[id_]
                stroke=2
                color=(0,255,0)
                cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            elif conf>100:
                font=cv2.FONT_HERSHEY_SIMPLEX
                stroke=2
                color=(0,255,0)
                cv2.putText(frame,"Unknown", (x,y), font, 1, color, stroke, cv2.LINE_AA)
            stroke=2 #độ dài của border
            width=x+w
            height=y+h
            cv2.rectangle(frame,(x,y),(width,height), (0,255,0), stroke)


        #hiển thị kết quả 
        cv2.imshow('frame',frame)

        #waitkey(20) đợi ít nhất 20ms, waitkey(0) chờ cho đến khi user nhấn phím bất kì
        #khi có 1 button dc nhấn thì waitkey() sẽ trả về 1 số nguyên 32 bit, 0xFF sẽ trả về số nhị phân 8 bit là 1 số có gt dưới 255
        #hàm ord(char) sẽ trả về 1 số có gt 0..255, nếu 0xFF trả về gt ascii bằng vs gt ascii của q thì thoát
        if cv2.waitKey(20) & 0xFF==ord('q'):
            break
    
    #Khi hoàn tất bắt hình thì hiển thị kết quả
    cap.release()
    cv2.destroyAllWindows()
