import cv2
import os
import csv
import time
cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)

face_detector = cv2.CascadeClassifier('C:/Users/admin/Documents/Python/Opencv_FaceDetect/haarcascade_frontalface_default.xml')

face_id = input('\n NHAP ID KHUON MAT <return>: ')
name = input('\n NHAP TEN: ')

print("\n [INFO] DANG KHOI TAO CAMERA VUI LONG CHO . . .")
count = 0
# Tạo thư mục để lưu ảnh huấn luyện
dataset_path = 'C:/Users/admin/Documents/Python/Opencv_FaceDetect/dataset'
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Vẽ 4 đoạn thẳng để tạo ra hình vuông tại 4 góc của khuôn mặt
        cv2.line(img, (x, y), (x + 30, y), (255, 0, 0), 2)        # Góc trái trên
        cv2.line(img, (x, y), (x, y + 30), (255, 0, 0), 2)
        
        cv2.line(img, (x + w, y), (x + w - 30, y), (255, 0, 0), 2)  # Góc phải trên
        cv2.line(img, (x + w, y), (x + w, y + 30), (255, 0, 0), 2)
        
        cv2.line(img, (x, y + h), (x + 30, y + h), (255, 0, 0), 2)  # Góc trái dưới
        cv2.line(img, (x, y + h), (x, y + h - 30), (255, 0, 0), 2)
        
        cv2.line(img, (x + w, y + h), (x + w - 30, y + h), (255, 0, 0), 2)  # Góc phải dưới
        cv2.line(img, (x + w, y + h), (x + w, y + h - 30), (255, 0, 0), 2)

        cv2.imshow('image', img)
        
        if count == 0:
            for i in range(5, 0, -1):
                print("Chuan bi chup anh sau {} giay...".format(i))
                time.sleep(1)
        
        count += 1
        cv2.imwrite(os.path.join(dataset_path, "User." + str(face_id) + '.' + str(count) + ".jpg"), gray[y:y + h, x:x + w])
        
        print("Da chup anh thu {} thanh cong".format(count))
        
        if count >= 20:
            print("Da chup {} anh thanh cong".format(count))
            break

    k = cv2.waitKey(100) & 0xff
    if k == 27 or count >= 20:
        break

print("\n [INFO] GHI THONG TIN VAO FILE CSV...")
# Ghi thông tin vào tệp CSV
csv_file = 'C:/Users/admin/Documents/Python/Opencv_FaceDetect/info/info.csv'
with open(csv_file, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([face_id, name])

print("\n [INFO] THOAT")
cam.release()
cv2.destroyAllWindows()
