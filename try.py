import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
from pylibdmtx.pylibdmtx import decode

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN

while True:
    _, frame = cap.read()
    frames = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    decodedObjects = pyzbar.decode(frames)
    #decodedObjects1 = decode(frame)
    
    for obj in decodedObjects:
    #for obj in (decodedObjects or decodedObjects1):
        print("Data", obj.data)
        cv2.putText(frame, str(obj.data), (50, 50), font, 2,(255, 0, 0), 3)
        
    

    cv2.imshow("OUTPUT", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
