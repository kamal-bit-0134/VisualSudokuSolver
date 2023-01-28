import numpy as np

print("Starting up")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utlis import *
import sudukoSolver
import cv2
import numpy as np
import tensorflow as tf
import keras

pathImage = 'images/1_su.jpg'
# pathImage = 'images/sudo_2.jpg'

# pathImage = 'images/w.png'


heightImg = 450
widthImg = 450
model = intializePredictionModel()   # LOAD the CNN Model

# Getting the image
img = cv2.imread(pathImage)


img = cv2.resize(img,(widthImg,heightImg)) # Resizing for the squares 450,450
imgBlank = np.zeros((heightImg,widthImg,3),np.uint8)  # Black image
imgThreshold = preProcess(img)

# Finding the contours
imgContours = img.copy()
imgBigContour = img.copy()
contours,hierarch = cv2.findContours(imgThreshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # Saving sll the contours
cv2.drawContours(imgContours, contours,-1,(0,255,0),3) # Draw the contours with a good separating color

# Biggest Contour
biggest , maxArea = biggestContour(contours) #The biggest contour
# print(f"biggest corner point {biggest}")

if biggest.size != 0:
    biggest = reorder(biggest)
    print(biggest)
    cv2.drawContours(imgBigContour,biggest,-1,(0,255,0),15) #Draw the biggest contour
    pts1 = np.float32(biggest) # Prepare points for warp
    pts2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2) # GER
    imgWarpColored = cv2.warpPerspective(img,matrix,(widthImg,heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)


# Splitting the image for the inner contours
# imgSolvedDigits = imgBlank.copy()
imgSolvedDigits = imgBlank.copy()
boxes = splitBoxes(imgWarpColored)
print(len(boxes))
numbers = getPrediction(boxes,model)
# print(numbers[0])
print(numbers)
# imgDetectedDigits = displayNumbers(imgDetectedDigits,numbers,color = (255,0,255))


# Converting to 9*9 matrix
temp = np.reshape(numbers,(9,9))
print(temp)
print("\t\t")
sudukoSolver.solve_f(temp)



cv2.imshow('img',img)
cv2.imshow('threshold',imgThreshold)
cv2.imshow('imgContours',imgContours)
cv2.imshow('imgBigContour',imgBigContour)
cv2.imshow('imgWrapColored',imgWarpColored)
cv2.imshow('Box 1',boxes[3])
cv2.imshow('Box 1',boxes[45])
cv2.imshow('Box 1',boxes[0])

# print(temp[0][0])
# print(temp[0][1])
# print(temp[0][2])
# print(temp[0][3])
# print(temp[0][4])
# print(temp[0][5])
# print(temp[0][6])
# print(temp[0][7])
# print(temp[0][8])

cv2.waitKey(0)

