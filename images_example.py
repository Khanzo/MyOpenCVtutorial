# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:05:41 2019

@author: Radik
"""

import cv2
import numpy as np

def imageread():
    image = cv2.imread("images/leniv.jpg")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     
    cv2.imshow("Gray leniv", gray_image)
    cv2.imshow("leniv", image)
    
    cv2.imwrite("mages/gray_leniv.jpg", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def drawingimage():  
    image = cv2.imread("images/leniv.jpg")
    roi = image[100: 280, 150: 320]
    shape = image.shape
    print(shape)
     
    blue = (255, 0, 0)
    red = (0, 0, 255)
    green = (0, 255, 0)
    violet = (180, 0, 180)
    yellow = (0, 180, 180)
    white = (255, 255, 255)
     
    cv2.line(image, (150, 30), (450, 35), blue, thickness=5)
    cv2.circle(image, (340, 205), 23, red, -1)
    cv2.rectangle(image, (250, 60), (450, 95), green, -1)
    cv2.ellipse(image, (350, 150), (80, 20), 5, 0, 360, violet, -1)
    points = np.array([[[140, 230], [380, 230], [320, 250], [250, 280]]], np.int32)
    cv2.polylines(image, [points], True, yellow, thickness=3)
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(image, "Leniv", (100, 300), font, 4, white)
    cv2.imshow("Roi", roi)
    cv2.imshow("leniv", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def histogram():
    from matplotlib import pyplot as plt
     
    img = cv2.imread("images/leniv.jpg")
    b, g, r = cv2.split(img)
     
    cv2.imshow("img", img)
    cv2.imshow("b", b)
    cv2.imshow("g", g)
    cv2.imshow("r", r)     
     
    plt.hist(b.ravel(), 256, [0, 256])
    plt.hist(g.ravel(), 256, [0, 256])
    plt.hist(r.ravel(), 256, [0, 256])
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def edge():
    img = cv2.imread("images/leniv.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (11, 11), 0)
     
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
     
    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
     
    canny = cv2.Canny(img, 100, 150)
     
    cv2.imshow("Image", img)
    cv2.imshow("Sobelx", sobelx)
    cv2.imshow("Sobely", sobely)
    cv2.imshow("Laplacian", laplacian)
    cv2.imshow("Canny", canny)
     
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def blur():
    img = cv2.imread("images/leniv.jpg") 
    averaging = cv2.blur(img, (21, 21))
    gaussian = cv2.GaussianBlur(img, (21, 21), 0)
    median = cv2.medianBlur(img, 5)
    bilateral = cv2.bilateralFilter(img, 9, 350, 350)
     
    cv2.imshow("Original image", img)
    cv2.imshow("Averaging", averaging)
    cv2.imshow("Gaussian", gaussian)
    cv2.imshow("Median", median)
    cv2.imshow("Bilateral", bilateral)
     
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def geometric_transform():
    img = cv2.imread("images/leniv.jpg")
    rows, cols, ch = img.shape
     
    print("Height: ", rows)
    print("Width: ", cols)
     
    scaled_img = cv2.resize(img, None, fx=1/2, fy=1/2)
     
    matrix_t = np.float32([[1, 0, -100], [0, 1, -30]])
    translated_img = cv2.warpAffine(img, matrix_t, (cols, rows))
     
    matrix_r = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 0.5)
    rotated_img = cv2.warpAffine(img, matrix_r, (cols, rows))
     
    cv2.imshow("Original image", img)
    cv2.imshow("Scaled image", scaled_img)
    cv2.imshow("Translated image", translated_img)
    cv2.imshow("Rotated image", rotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def perspective_transform():
    frame = cv2.imread("images/leniv.jpg")
    cv2.circle(frame, (155, 120), 5, (0, 0, 255), -1)
    cv2.circle(frame, (480, 120), 5, (0, 0, 255), -1)
    cv2.circle(frame, (20, 475), 5, (0, 0, 255), -1)
    cv2.circle(frame, (620, 475), 5, (0, 0, 255), -1)
 
    pts1 = np.float32([[155, 120], [480, 120], [20, 475], [620, 475]])
    pts2 = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
 
    result = cv2.warpPerspective(frame, matrix, (500, 600)) 
 
    cv2.imshow("Frame", frame)
    cv2.imshow("Perspective transformation", result)
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def morphological_transform():
    img = cv2.imread("images/balls.jpg", cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)
     
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(mask, kernel)
    erosion = cv2.erode(mask, kernel, iterations=6)
     
    cv2.imshow("Image", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Dilation", dilation)
    cv2.imshow("Erosion", erosion)
     
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
def affine_transform():
    img = cv2.imread("images/grid.jpg")
    rows, cols, ch = img.shape
     
    cv2.circle(img, (83, 90), 5, (0, 0, 255), -1)
    cv2.circle(img, (447, 90), 5, (0, 0, 255), -1)
    cv2.circle(img, (83, 472), 5, (0, 0, 255), -1)
     
    pts1 = np.float32([[83, 90], [447, 90], [83, 472]])
    pts2 = np.float32([[0, 0], [447, 90], [150, 472]])
     
    matrix = cv2.getAffineTransform(pts1, pts2)
    result = cv2.warpAffine(img, matrix, (cols, rows))
     
    cv2.imshow("Image", img)
    cv2.imshow("Affine transformation", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def adaptive_threshold():
    img = cv2.imread("images/leniv.jpg")     
    _, threshold = cv2.threshold(img, 155, 255, cv2.THRESH_BINARY)     
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
    mean_c = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 12)
    gaus = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 12)
     
    cv2.imshow("Img", img)
    cv2.imshow("Binary threshold", threshold)
    cv2.imshow("Mean C", mean_c)
    cv2.imshow("Gaussian", gaus)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    
def threshold():
    
    def nothing(x):
        pass
 
    img = cv2.imread("images/leniv.jpg", cv2.IMREAD_GRAYSCALE)
    cv2.namedWindow("Image")
    cv2.createTrackbar("Threshold value", "Image", 128, 255, nothing)
          
    while True:
        value_threshold = cv2.getTrackbarPos("Threshold value", "Image")
        _, threshold_binary = cv2.threshold(img, value_threshold, 255, cv2.THRESH_BINARY)
        _, threshold_binary_inv = cv2.threshold(img, value_threshold, 255, cv2.THRESH_BINARY_INV)
        _, threshold_trunc = cv2.threshold(img, value_threshold, 255, cv2.THRESH_TRUNC)
        _, threshold_to_zero = cv2.threshold(img, value_threshold, 255, cv2.THRESH_TOZERO)
        _, threshold_to_zero_inv = cv2.threshold(img, value_threshold, 255, cv2.THRESH_TOZERO_INV)
        
        cv2.imshow("Image", img)
        cv2.imshow("th binary", threshold_binary)
        cv2.imshow("th binary inv", threshold_binary_inv)
        cv2.imshow("th trunc", threshold_trunc)
        cv2.imshow("th to zero", threshold_to_zero)
        cv2.imshow("th to zero inv", threshold_to_zero_inv)
     
        key = cv2.waitKey(100)
        if key == 27:
            break
     
    cv2.destroyAllWindows()
    
def two_image():
    img1 = cv2.imread("images/water.jpg")
    img2 = cv2.imread("images/leniv_black.jpg")
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
     
    print(img1[0, 0])
    print(img2[0, 0])
     
    weighted = cv2.addWeighted(img1, 1, img2, 0.5, 0)
    #ret, mask = cv2.threshold(img2_gray, 252, 255, cv2.THRESH_BINARY_INV)
    ret, mask = cv2.threshold(img2_gray, 240, 255, cv2.THRESH_BINARY)
    
    mask_inv = cv2.bitwise_not(mask)
 
    water = cv2.bitwise_and(img1, img1, mask=mask)
    leniv = cv2.bitwise_and(img2, img2, mask=mask_inv)
    result = cv2.add(water, leniv)
     
    sum = cv2.add(img1, img2, mask=mask)
     
    cv2.imshow("sum", sum)
    cv2.imshow("threshold", mask)
    cv2.imshow("img2gray", img2_gray)
    cv2.imshow("weighted", weighted)
    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    cv2.imshow("mask inverse", mask_inv)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    
def bit_operartion():
    img1 = cv2.imread("images/drawing_1.png")
    img2 = cv2.imread("images/drawing_2.png")
     
    bit_and = cv2.bitwise_and(img2, img1)
    bit_or = cv2.bitwise_or(img2, img1)
    bit_xor = cv2.bitwise_xor(img1, img2)
    bit_not = cv2.bitwise_not(img1)
    bit_not2 = cv2.bitwise_not(img2)     
     
    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
     
    cv2.imshow("bit_and", bit_and)
    cv2.imshow("bit_or", bit_or)
    cv2.imshow("bit_xor", bit_xor)
    cv2.imshow("bit_not", bit_not)
    cv2.imshow("bit_not2", bit_not2)     
     
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def template_matching():
    img = cv2.imread("images/leniv.jpg")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread("images/leniv_face.jpg", cv2.IMREAD_GRAYSCALE)
    w, h = template.shape[::-1]
     
    result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= 0.4)
     
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)
        
    cv2.imshow("img", img)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def lines_detection():
    img = cv2.imread("images/grid.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 75, 150)
     
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)
     
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
     
    cv2.imshow("Edges", edges)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def contur():
    frame = cv2.imread("images/leniv.jpg")
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
 
    lower_ = np.array([38, 86, 0])
    upper_ = np.array([93, 255, 255])
    mask = cv2.inRange(hsv, lower_, upper_)
 
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
 
    for contour in contours:
        area = cv2.contourArea(contour)
 
        if area > 5000:
            cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
 
 
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def cornes():
    img = cv2.imread("images/grid.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
    corners = cv2.goodFeaturesToTrack(gray, 150, 0.8, 50)
    corners = np.int0(corners)
     
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
     
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def back_project():
    original_image = cv2.imread("images/leniv.jpg")
    hsv_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
     
    roi = cv2.imread("images/leniv_ground.jpg")
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
     
    hue, saturation, value = cv2.split(hsv_roi)
     
     
    # Histogram ROI
    roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    mask = cv2.calcBackProject([hsv_original], [0, 1], roi_hist, [0, 180, 0, 256], 1)
     
    # Filtering remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.filter2D(mask, -1, kernel)
    _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
     
    mask = cv2.merge((mask, mask, mask))
    result = cv2.bitwise_and(original_image, mask)
     
    cv2.imshow("Mask", mask)
    cv2.imshow("Original image", original_image)
    cv2.imshow("Result", result)
    cv2.imshow("Roi", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def shapes():
    img = cv2.imread("images/shapes.jpg", cv2.IMREAD_GRAYSCALE)
    _, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     
    font = cv2.FONT_HERSHEY_COMPLEX
     
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        cv2.drawContours(img, [approx], 0, (0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
     
        if len(approx) == 3:
            cv2.putText(img, "Triangle", (x, y), font, 1, (0))
        elif len(approx) == 4:
            cv2.putText(img, "Rectangle", (x, y), font, 1, (0))
        elif len(approx) == 5:
            cv2.putText(img, "Pentagon", (x, y), font, 1, (0))
        elif 6 < len(approx) < 15:
            cv2.putText(img, "Ellipse", (x, y), font, 1, (0))
        else:
            cv2.putText(img, "Circle", (x, y), font, 1, (0))
     
    cv2.imshow("shapes", img)
    cv2.imshow("Threshold", threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def fourier_transform():
    img = cv2.imread("images/leniv.jpg", cv2.IMREAD_GRAYSCALE)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
    img_and_magnitude = np.concatenate((img, magnitude_spectrum), axis=1)
 
    cv2.imshow("fourier", img_and_magnitude)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def pyramid():
    img = cv2.imread("images/leniv.jpg")     
    # Gaussian Pyramid
    layer = img.copy()
    gaussian_pyramid = [layer]
    for i in range(6):
        layer = cv2.pyrDown(layer)
        gaussian_pyramid.append(layer)     
    # Laplacian Pyramid
    layer = gaussian_pyramid[5]
    cv2.imshow("6", layer)
    laplacian_pyramid = [layer]
    for i in range(5, 0, -1):
        size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
        laplacian_pyramid.append(laplacian)
        cv2.imshow(str(i), laplacian)
     
    cv2.imshow("Original image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def reconstructed_image():
    img = cv2.imread("images/leniv.jpg") 
    # Gaussian Pyramid
    layer = img.copy()
    gaussian_pyramid = [layer]
    for i in range(6):
        layer = cv2.pyrDown(layer)
        gaussian_pyramid.append(layer)
     
    # Laplacian Pyramid
    layer = gaussian_pyramid[5]
    laplacian_pyramid = [layer]
    for i in range(5, 0, -1):
        size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
        laplacian_pyramid.append(laplacian)     
     
    reconstructed_image = laplacian_pyramid[0]
    for i in range(1, 6):
        size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
        reconstructed_image = cv2.pyrUp(reconstructed_image, dstsize=size)
        reconstructed_image = cv2.add(reconstructed_image, laplacian_pyramid[i])
        cv2.imshow(str(i), reconstructed_image)
     
    cv2.imshow("original", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def feature_detect():
    img = cv2.imread("images/leniv.jpg", cv2.IMREAD_GRAYSCALE)
 
    #sift = cv2.SIFT_create()
    #surf = cv2.SURF_create()
     
    orb = cv2.ORB_create(nfeatures=1500)     
    keypoints, descriptors = orb.detectAndCompute(img, None)     
    img = cv2.drawKeypoints(img, keypoints, None)
     
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def matching_feature_brutforce():
    img1 = cv2.imread("images/leniv.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("images/leniv_rotate.jpg", cv2.IMREAD_GRAYSCALE)
     
    # ORB Detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
     
    # Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
     
    matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
     
    cv2.imshow("Img1", img1)
    cv2.imshow("Img2", img2)
    cv2.imshow("Matching result", matching_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def compare_image():
    def xrange(x):
        return iter(range(x))
    
    from matplotlib import pyplot as plt
    img1 = cv2.imread("images/leniv.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("images/leniv_rotate.jpg", cv2.IMREAD_GRAYSCALE)
     
    # ORB Detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    #FLANN_INDEX_KDTREE = 0
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
    search_params = dict(checks=50)   # or pass empty dictionary
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    matches = flann.knnMatch(des1,des2,k=2)
    matchesMask = [[0,0] for i in xrange(len(matches))]
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)
    
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    
    plt.imshow(img3,),plt.show()
    
    
#imageread()
#drawingimage()
#histogram()
#edge()
#blur()
#geometric_transform()
#affine_transform()
#adaptive_threshold()
#threshold()    
#morphological_transform()
#bit_operartion()
#perspective_transform()
#two_image()
#template_matching()
#lines_detection()
#contur()
#cornes()
#back_project()
#shapes()
#fourier_transform()
#pyramid()
#reconstructed_image()
#feature_detect()
#matching_feature_brutforce()
    
compare_image()