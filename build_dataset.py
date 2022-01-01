import cv2
import os

def build_dataset(name,id):
    face_cascade=cv2.CascadeClassifier('face-recog\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')

    cap=cv2.VideoCapture(0)
    total=0
    #tạo thư mục tên
    dir="dataset/"+name+"_"+id
    if not os.path.exists(dir):
        os.mkdir(dir)
    while (True):
        #bắt từng khung hình
        ret, frame =cap.read()

        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x,y,w,h) in faces:
            roi_gray=gray[y:y+h,x:x+w]
            color=(0,255,0)
            stroke=2 #độ dài của border
            width=x+w
            height=y+h
            cv2.rectangle(frame,(x,y),(width,height), color, stroke)

        #hiển thị kết quả 
        key=cv2.waitKey(1) & 0xFF
        cv2.imshow('frame',frame)
        if key==ord('k'):
            cv2.imwrite(dir+"/"+str(total)+".png", frame)
            total+=1
            
        elif key==ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()