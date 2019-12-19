# first neural network with keras tutorial
import cv2
import os
import numpy as np
from math import copysign, log10
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from pyefd import elliptic_fourier_descriptors
from sklearn.preprocessing import StandardScaler
import imutils
from sklearn import preprocessing
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
# load the dataset


def dct(img):
    imf = np.float32(img)/255.0  # float conversion/scale
    dcts = cv2.dct(imf)           # the dct
    return dcts


def PreProcess(im, option):
    img = cv2.imread(im)

    if option == 0:
        return_img = img
    if option == 1:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return_img = gray
    elif option == 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return_img = binary
    # return_img = np.pad(return_img,((1,1),(1,1)),mode='constant',constant_values=255)
    return return_img


def baseline_model():
    model = Sequential()
    model.add(Dense(90, input_dim=2500, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # compile the keras model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


folders = ['alif', 'baa', 'taa', 'seh', 'jiim',
           'haa', 'khaa', 'daal',  'raa', 'siin', 'shiin', 'saad', 'daad', 'tah', 'een', 'ghin', 'faa', 'qaaf',
           'kaaf', 'laam', 'miim', 'noon', 'heh', 'waaw', 'laamalif', 'zaal', 'zeen', 'zaa', 'yaa2'
           ]


dataset = []

'''
labels = np.zeros(len(folders)*200)
labels[0:199] = 1
labels[200:399] = 0
'''
label = []

for folder in folders:
    dirName = os.path.join("C:/Users\medo\Desktop\pattern2\dataset", folder)
    print(dirName)

    listOfFiles = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(dirName):
        for file in f:

            listOfFiles.append(os.path.join(r, file))

    # Print the files
    for i, elem in enumerate(listOfFiles):
        if i > 199:
            break
        gray = PreProcess(elem, 1)
        binary = PreProcess(elem, 2)
        dataset.append(gray)
        label.append(folder)


feature_vector = []
for i in range(len(dataset)):
    feature_vector.append(dct(dataset[i]).flatten())

'''
scaler = StandardScaler()
scaler.fit(feature_vector)
scaler.transform(feature_vector)
'''
feature_matrix = np.array(feature_vector)

x = pd.DataFrame(feature_matrix)
y = pd.Series(label)


X, x_test, Y, y_test = train_test_split(x, y, test_size=0.2)


final_labels = {}
data = []

for i in range(len(folders)):
    final_labels[folders[i]] = i


for i in range(len(Y)):
    data.append(final_labels[y[i]])


data = array(data)
print(data)
# one hot encode
encoded = to_categorical(data)
print(encoded.shape)

'''
output_labels = {}
for i, label in enumerate(final_labels):
    output_labels[label] = integer_encoded[i]


print(onehot_encoded[output_labels['alif']])
print(onehot_encoded)
'''
