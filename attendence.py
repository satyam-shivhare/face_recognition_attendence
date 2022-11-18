import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'imagesAttendence'
images = []
classNames = []

myList = os.listdir(path)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

print(classNames)

def find_encodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode= face_recognition.face_encodings(img)[0]
        encodelist.append(encode)

    return encodelist

def mark_attendance(name):
    with open('Attendence.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}' )

encodeListKnown = find_encodings(images)
print('encoding complete')

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgSmall = cv2.resize(img, (0,0),None, 0.25,0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(imgSmall)
    encodingsCurrentFrame = face_recognition.face_encodings(imgSmall,facesCurrentFrame)

    for encodeFace,faceLoc in zip(encodingsCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 =  y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img, name, (x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX, 1 ,(255,255,255),2)
            mark_attendance(name)


    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
