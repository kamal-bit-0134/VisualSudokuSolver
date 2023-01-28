import cv2
import numpy as np
from tensorflow.keras.models import load_model

# preProcess for threshold
def preProcess(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)  #Blurring th img
    imgThreshold = cv2.adaptiveThreshold(imgBlur,255,1,1,11,2)
    return imgThreshold

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if(area>50):
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            if ( area > max_area and len(approx)==4 ):
                biggest = approx
                max_area = area
    return biggest, max_area

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def splitBoxes(img):
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes

def intializePredictionModel():
    model = load_model('MyModel.h5')
    return model

def getPrediction(boxes,model):
    result = []
    for image in boxes:
        # Preparing the image
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4,4:img.shape[1] -4]
        img = cv2.resize(img,(28,28))
        img = img/255
        img = img.reshape(1,28,28,1)
        # Get Prediction
        predictions = model.predict(img)
        # ClassIndex = model.predct_classes(img)
        classIndex = np.argmax(predictions,axis=-1)
        probabilityValue = np.argmax(predictions)
        print(classIndex,probabilityValue)
        # Saving the result
        if probabilityValue > 0.8:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result

def displayNumbers(img,numbers,color = (0,255,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range(0,9):
            if(numbers[(y+9)+x]!=0):
                cv2.putText(img,str(numbers[(y*9)+x]),(x+secW+int(secW/2)-10,int((y+0.8)*secH)),cv2.FONT_HERSHEY_COMPLEX)