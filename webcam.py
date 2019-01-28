# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 21:06:23 2019

@author: Radik
"""
import numpy as np
import cv2
  
def openwebcam():
    #cap = cv2.VideoCapture("video_file.mp4")  
    cap = cv2.VideoCapture(0)
      
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('Video', frame)    
        cv2.imshow('frame',gray)
        #cv2.imwrite('cam.png', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      
    cap.release()
    cv2.destroyAllWindows()
    
def trackbars():
    def nothing(x):
        pass 
    
    cap = cv2.VideoCapture(0)     
    cv2.namedWindow("frame")
    cv2.createTrackbar("test", "frame", 50, 500, nothing)
    cv2.createTrackbar("color/gray", "frame", 0, 1, nothing)
     
    while True:
        _, frame = cap.read()
     
        test = cv2.getTrackbarPos("test", "frame")
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame, str(test), (50, 150), font, 4, (0, 0, 255))
     
        s = cv2.getTrackbarPos("color/gray", "frame")
        if s == 0:
            pass
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
        cv2.imshow("frame", frame)
     
        if cv2.waitKey(1) & 0xFF == ord('q'):      
            break
     
    cap.release()
    cv2.destroyAllWindows()
    
def template_matching():
    cap = cv2.VideoCapture(0)
    template = cv2.imread("images/pen.png", cv2.IMREAD_GRAYSCALE)
    w, h = template.shape[::-1]
     
    while True:
        _, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
        res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= 0.7)
     
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)
     
        cv2.imshow("Frame", frame)
     
        if cv2.waitKey(1) & 0xFF == ord('q'):      
            break
     
    cap.release()
    cv2.destroyAllWindows()
    
def edge():
    cap = cv2.VideoCapture(0)
 
    while True:
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
     
        laplacian = cv2.Laplacian(blurred_frame, cv2.CV_64F)
        canny = cv2.Canny(blurred_frame, 100, 150)
     
        cv2.imshow("Frame", frame)
        cv2.imshow("Laplacian", laplacian)
        cv2.imshow("Canny", canny)
     
     
        if cv2.waitKey(1) & 0xFF == ord('q'):      
            break
        
    cap.release()
    cv2.destroyAllWindows()
        
def contur():
    cap = cv2.VideoCapture(0) 
    while True:
        _, frame = cap.read()
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
     
        lower_blue = np.array([38, 86, 0])
        upper_blue = np.array([121, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
     
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
     
        for contour in contours:
            area = cv2.contourArea(contour)
     
            if area > 5000:
                cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
     
     
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)
     
        if cv2.waitKey(1) & 0xFF == ord('q'):      
            break
     
    cap.release()
    cv2.destroyAllWindows()
    
def HSVdetectobject():
    def nothing(x):
        pass
 
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Trackbars")
     
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
     
     
    while True:
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")
     
        lower_blue = np.array([l_h, l_s, l_v])
        upper_blue = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
     
        result = cv2.bitwise_and(frame, frame, mask=mask)
     
        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        cv2.imshow("result", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):      
            break
     
    cap.release()
    cv2.destroyAllWindows()
    
def lines_detect():
    #cap = cv2.VideoCapture("Cars - 1900.mp4")  
    #https://pixabay.com/ru/videos/автомобили-автомагистраль-скорость-1900/  
    cap = cv2.VideoCapture(0)
    while True:
        ret, orig_frame = cap.read()
        #if not ret:
        #    cap = cv2.VideoCapture("road_car_view.mp4")
        #    continue
     
        frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low_yellow = np.array([18, 94, 140])
        up_yellow = np.array([48, 255, 255])
        mask = cv2.inRange(hsv, low_yellow, up_yellow)
        edges = cv2.Canny(mask, 75, 150)
     
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
     
        cv2.imshow("frame", frame)
        cv2.imshow("edges", edges)
     
        if cv2.waitKey(1) & 0xFF == ord('q'):      
            break
    cap.release()
    cv2.destroyAllWindows()
    
def cornes():
    cap = cv2.VideoCapture(0)
 
    def nothing(x):
        pass
     
    cv2.namedWindow("Frame")
    cv2.createTrackbar("quality", "Frame", 1, 100, nothing)
     
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
        quality = cv2.getTrackbarPos("quality", "Frame")
        quality = quality / 100 if quality > 0 else 0.01
        corners = cv2.goodFeaturesToTrack(gray, 100, quality, 20)
     
        if corners is not None:
            corners = np.int0(corners)
     
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
     
        cv2.imshow("Frame", frame)
     
        if cv2.waitKey(1) & 0xFF == ord('q'):      
            break
     
    cap.release()
    cv2.destroyAllWindows()
    
def object_tracking():
    roi = cv2.imread("images/cover.jpg")
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
     
    cap = cv2.VideoCapture(0)     
    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    
    x = 300
    y = 305
    width = 100
    height = 115
    
    """
    _, first_frame = video.read()
    
    roi = first_frame[y: y + height, x: x + width]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)     
    """
    while True:
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
     
        ret, track_window = cv2.CamShift(mask, (x, y, width, height), term_criteria)
     
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
        
        """
        _, frame = video.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
     
        _, track_window = cv2.meanShift(mask, (x, y, width, height), term_criteria)
        x, y, w, h = track_window
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        """
     
        cv2.imshow("mask", mask)
        cv2.imshow("Frame", frame)
     
        if cv2.waitKey(1) & 0xFF == ord('q'):      
            break
     
    cap.release()
    cv2.destroyAllWindows()

def lucas_kanade_tracking():
    cap = cv2.VideoCapture(0)
     
    # Create old frame
    _, frame = cap.read()
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
    # Lucas kanade params
    lk_params = dict(winSize = (15, 15),
                     maxLevel = 4,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                                 10, 0.03))
     
    # Mouse function
    def select_point(event, x, y, flags, params):
        global point, point_selected, old_points
        if event == cv2.EVENT_LBUTTONDOWN:
            point = (x, y)
            point_selected = True
            old_points = np.array([[x, y]], dtype=np.float32)
     
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", select_point)
     
    point_selected = False
    point = ()
    old_points = np.array([[]])
    while True:
        _, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
        if point_selected is True:
            cv2.circle(frame, point, 5, (0, 0, 255), 2)
            
            new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, 
                                                                 gray_frame, 
                                                                 old_points, 
                                                                 None, 
                                                                 **lk_params)
            old_gray = gray_frame.copy()
            old_points = new_points
     
            x, y = new_points.ravel()
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)   
     
        cv2.imshow("Frame", frame)
     
        if cv2.waitKey(1) & 0xFF == ord('q'):      
            break
     
    cap.release()
    cv2.destroyAllWindows()
    
def subtractor_first_farme():
    cap = cv2.VideoCapture("Cars - 1900.mp4")    
    #https://pixabay.com/ru/videos/автомобили-автомагистраль-скорость-1900/  
    _, first_frame = cap.read()
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)
     
    while True:
        _, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
     
        difference = cv2.absdiff(first_gray, gray_frame)
        _, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
     
        cv2.imshow("First frame", first_frame)
        cv2.imshow("Frame", frame)
        cv2.imshow("difference", difference)
     
        if cv2.waitKey(1) & 0xFF == ord('q'):      
            break  
    cap.release()
    cv2.destroyAllWindows()
    
def subtractor_MOG2():
    cap = cv2.VideoCapture("Cars - 1900.mp4")  
    #https://pixabay.com/ru/videos/автомобили-автомагистраль-скорость-1900/   
    subtractor = cv2.createBackgroundSubtractorMOG2(history=20, 
                                                    varThreshold=25, 
                                                    detectShadows=True)     
    while True:
        _, frame = cap.read()     
        mask = subtractor.apply(frame)     
        cv2.imshow("Frame", frame)
        cv2.imshow("mask", mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):      
            break  
     
    cap.release()
    cv2.destroyAllWindows()
    
    
#openwebcam()
#trackbars()  
#template_matching()
#subtractor_first_farme()
#subtractor_MOG2()
#object_tracking()
#cornes()
#lines_detect()
#HSVdetectobject()
#contur()
#edge()