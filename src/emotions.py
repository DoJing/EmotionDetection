import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from src.Model import EmotionNet
# Find haar cascade to draw bounding box around face
frame = cv2.imread("/home/dojing/图片/641.jpg")
facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
plt.figure("Image")  # 图像窗口名称

net = EmotionNet('model.h5')

for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
    prediction = net.predict(cropped_img)

    cv2.putText(frame, prediction, (x+20, y+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    print(prediction)
plt.imshow(frame[:,:,(2,1,0)])
plt.show()
