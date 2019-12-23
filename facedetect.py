# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:47:11 2019

@author: Radik
"""

import cv2

def facedetect_haar():
    faceCascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
    vs = cv2.VideoCapture(0)
    #static_back = None

    while True:
        ret, frame = vs.read()
        
        if frame is None:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #faces = faceCascade.detectMultiScale(frame)
        #gray = cv2.GaussianBlur(gray, (21, 21), 0)

        #if static_back is None:
        #    static_back = gray
        #    continue

        #diff_frame = cv2.absdiff(static_back, gray)

        # If change in between static background and
        # current frame is greater than 30 it will show white color(255)
        #thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        #thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
        #frame = thresh_frame

        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

        cv2.imshow("Video", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    vs.release()
    cv2.destroyAllWindows()

def eyes_detect_haar():
    faceCascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
    eyeCascade= cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')
    vs = cv2.VideoCapture(0)
    
    while True:
        ret, frame = vs.read()
        
        if frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = faceCascade.detectMultiScale(frame)

        for (x,y,w,h) in faces:
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eyeCascade.detectMultiScale(roi_gray)
            
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                
        cv2.imshow("Video", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    vs.release()
    cv2.destroyAllWindows()

def smile_detect_harr():
    faceCascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
    smileCascade= cv2.CascadeClassifier('haarcascade/haarcascade_smile.xml')
    vs = cv2.VideoCapture(0)

    while True:
        ret, frame = vs.read()
        
        if frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = faceCascade.detectMultiScale(frame, scaleFactor=1.05,
                                          minNeighbors=5,
                                          minSize=(45, 45))

        for (x,y,w,h) in faces:
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            face_gray = gray[y:y+h, x:x+w]
            face_color = frame[y:y+h, x:x+w]
            smiles = smileCascade.detectMultiScale(face_gray,
                                             scaleFactor=1.7,
                                             minNeighbors=3,
                                             minSize=(15, 15))
            for (ex,ey,ew,eh) in smiles:
                cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)

        cv2.imshow("Video", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    vs.release()
    cv2.destroyAllWindows()

#facedetect_haar()
#smile_detect_harr()
#eyes_detect_haar()