import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.optimizers import adam_v2
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle
# Initializations
path = 'myData'
images = []
classNo = []
testRatio = 0.2
validationRatio = 0.2
imageDimensions = (32,32,3)

mylist = os.listdir(path)
# print(mylist)
print("Total no. of classes")
print(len(mylist))
noOfClasses = len(mylist)
print("Reading images into list")
for x in range (0,noOfClasses):
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg,(32,32))
        images.append(curImg)
        classNo.append(x)
    print(x,end=" ")
print(" ")

# print(len(images))
# print(len(classNo))
images = np.array(images)
classNo = np.array(classNo)

# Displays classes ,size(x,y), colors(3 for BGR)
# print(images.shape)
# print(classNo.shape)

# Splitting the dataset
X_train,X_test,y_train,y_test = train_test_split(images,classNo,test_size=testRatio)
X_train,X_validtion,y_train,y_validation = train_test_split(X_train,y_train,test_size=validationRatio)

print("X_train    X_test      X_valdation")
print(X_train.shape)
print(X_test.shape)
print(X_validtion.shape)
# print(y_validation.shape)

numOfSamples = []

for x in range (0,noOfClasses):
    # print(len(np.where(y_train == x)[0]))
    numOfSamples.append(len(np.where(y_train == x)[0]))
print(numOfSamples)

plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClasses),numOfSamples)
plt.title("No of Images for each Class")
plt.xlabel("Class ID")
plt.ylabel("No. of Images")
plt.show()

# Pre processing image
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

# img = preProcessing(X_train[30])
# img = cv2.resize(img,(300,300))
# cv2.imshow('PreProcessed Image',img)
# cv2.waitKey(0)
# print(X_train[30].shape)
X_train = np.array(list(map(preProcessing,X_train)))
X_test = np.array(list(map(preProcessing,X_test)))
X_validtion = np.array(list(map(preProcessing,X_validtion)))

# print(X_train[30].shape)

# print(X_train.shape)
# Adding depth for CNN
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validtion = X_validtion.reshape(X_validtion.shape[0],X_validtion.shape[1],X_validtion.shape[2],1)
# print(X_train.shape) checking for the depth 1(6502,32,32,1)

# Generate images to augamated
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,zoom_range=0.2,
                             shear_range=0.1,rotation_range=10)

dataGen.fit(X_train)

y_train = to_categorical(y_train,noOfClasses)
# y_train = to_categorical(y_train)
y_test = to_categorical(y_test,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)
# y_validation = to_categorical(y_validation)

def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNodes = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1,
                      input_shape=(imageDimensions[0],imageDimensions[1],1)
                      ,activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1,activation='relu')))
    model.add(MaxPooling2D(pool_size = sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2,activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2,activation='relu')))
    model.add(MaxPooling2D(pool_size = sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses,activation='softmax'))
    model.compile(optimizer= 'adam',loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())

batchSizeVal = 50
epochsVal = 10
stepsPerEpochVal = 200
#
# history = model.fit_generator(dataGen.flow(X_train,y_train,batch_size=batchSizeVal),
#                     steps_per_epoch=stepsPerEpochVal,
#                     epochs=epochsVal,
#                     validation_data=(X_validtion,y_validation),
#                     shuffle=1)
history = model.fit(X_train,y_train,batch_size=batchSizeVal,validation_data=(X_validtion,y_validation),epochs=5,shuffle=1)
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training','Validation'])
plt.title('accuracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test,y_test,verbose=0)
print("Test score = "+ str(score[0]))
print("Test Accuracy = "+str(score[1]))

# model.save('Temp_model.h5')
pickle_out = open("model_trained.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()
