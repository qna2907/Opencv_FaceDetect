import cv2
import numpy as np
import os
import csv

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:/Users/admin/Documents/Python/Opencv_FaceDetect/trainer/trainer.yml')
cascadePath = "C:/Users/admin/Documents/Python/Opencv_FaceDetect/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

id_to_name = {}  # Tạo một từ điển để ánh xạ id sang tên

# Đọc thông tin từ tệp CSV và lưu vào từ điển id_to_name
csv_file = 'C:/Users/admin/Documents/Python/Opencv_FaceDetect/info/info.csv'
if os.path.exists(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Bỏ qua dòng tiêu đề
        for row in reader:
            if row:
                id_to_name[int(row[0])] = row[1]

cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray, 
        scaleFactor=1.2, 
        minNeighbors=5, 
        minSize=(int(minW), int(minH))
        )

    for (x, y, w, h) in faces:
        cv2.line(img, (x, y), (x + int(w/4), y), (0, 255, 0), 2)  # Vẽ đoạn ngang trên cùng
        cv2.line(img, (x, y), (x, y + int(h/4)), (0, 255, 0), 2)  # Vẽ đoạn dọc bên trái

        cv2.line(img, (x + w, y), (x + w - int(w/4), y), (0, 255, 0), 2)  # Vẽ đoạn ngang trên cùng
        cv2.line(img, (x + w, y), (x + w, y + int(h/4)), (0, 255, 0), 2)  # Vẽ đoạn dọc bên phải

        cv2.line(img, (x, y + h), (x + int(w/4), y + h), (0, 255, 0), 2)  # Vẽ đoạn ngang dưới cùng
        cv2.line(img, (x, y + h), (x, y + h - int(h/4)), (0, 255, 0), 2)  # Vẽ đoạn dọc bên trái

        cv2.line(img, (x + w, y + h), (x + w - int(w/4), y + h), (0, 255, 0), 2)  # Vẽ đoạn ngang dưới cùng
        cv2.line(img, (x + w, y + h), (x + w, y + h - int(h/4)), (0, 255, 0), 2)  # Vẽ đoạn dọc bên phải


        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        id, confidence = recognizer.predict(roi_gray)

        if confidence < 100:
            if id in id_to_name:
                id = id_to_name[id]
            confidence = " {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = " {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

    cv2.imshow('NHAN DIEN KHUON MAT', img)

    k = cv2.waitKey(10) & 0xff  # Nhấn 'ESC' để thoát
    if k == 27:
        break

print("\n [INFO] THOAT")
cam.release()
cv2.destroyAllWindows()

