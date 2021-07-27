# Face Detection


import numpy as np
import matplotlib.pyplot as plt
import cv2 # ----> opencv

# read an image
img = plt.imread("/content/drive/MyDrive/Dataset/group.jpg")

type(img)

img.shape

img.ndim

plt.imshow(img)

model = cv2.CascadeClassifier("/content/drive/MyDrive/Dataset/haarcascade_frontalface_default.xml")

all_faces = model.detectMultiScale(img,1.5)

all_faces

x, y, w, h = all_faces[0]

print(x)
print(y)
print(w)
print(h)

img = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 0), 3 )

plt.imshow(img)

for ele in all_faces:
  x,y,w,h = ele
  img = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 0), 2)

plt.imshow(img)
