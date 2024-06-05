import cv2
import numpy as np
import time

FaceClassifier  = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

videoCam = cv2.VideoCapture(0)

if not videoCam.isOpened():
    print("The camera is not accessible")
    exit()
button = False
while (button == False):
    ret, framework = videoCam.read()

    if ret == True:
        gray = cv2.cvtColor(framework, cv2.COLOR_BGR2GRAY)
        Face = FaceClassifier.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 2)

        for (x, y, w, h) in Face:
            cv2.rectangle(framework, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        
        text = "Number of Faces Detected = " + str(len(Face))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(framework, text, (0, 30), font, 1, (255, 0, 0), 1)

        cv2.imshow("Results", framework)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            button = True
            break


videoCam.release()
cv2.destroyAllWindows()