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


def efd_feature(contour):
    coeffs = elliptic_fourier_descriptors(
        contour, order=10, normalize=True)
    return coeffs.flatten()[3:]


def efd(im):
    contours, hierarchy = cv2.findContours(
        im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    coeffs = []
    for cnt in contours:
        efd_feature(cnt)
        break


def HuMoments(img):
    moments = cv2.moments(img)
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)
    for i in range(0, 7):
        huMoments[i] = -1 * \
            copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
    return huMoments


def PreProcess(im):
    img = cv2.imread(im)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return binary


'''
folders = ['alif', 'baa', 'taa', 'seh', 'jiim', 'haa', 'khaa', 'daal', 'zaal', 'raa', 'zeen', 'siin', 'shiin',
           'saad', 'daad', 'tah', 'zaa', 'een', 'ghin', 'faa', 'qaaf', 'kaaf', 'laam', 'miim', 'noon', 'heh', 'waaw', 'yaa',
           'laamalif', 'yaa2']
'''
folders = ['alif', 'baa', 'taa', 'seh', 'jiim', 'haa', 'khaa', 'daal']
dataset = []
labels = np.zeros(8000)
labels[1000:1999] = 1
labels[2000:2999] = 2
labels[3000:3999] = 3
labels[4000:4999] = 4
labels[5000:5999] = 5
labels[6000:6999] = 6
labels[7000:7999] = 7

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
        if i > 999:
            break
        binary = PreProcess(elem)
        dataset.append(binary)


print(len(dataset))
feature_vector = []
for i in range(len(dataset)):
    feature_vector.append(HuMoments(dataset[i]).flatten())

scaler = StandardScaler()
scaler.fit(feature_vector)
scaler.transform(feature_vector)

feature_matrix = np.array(feature_vector)

x = pd.DataFrame(feature_matrix)
y = pd.Series(labels)


print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

svm = SVC(kernel="linear", probability=True)
svm.fit(x_train, y_train)

gnp = GaussianNB()
gnp.fit(x_train, y_train)

y_predict = svm.predict(x_test)
y_predict2 = gnp.predict(x_test)

accuracy_svm = accuracy_score(y_test, y_predict)
accuracy_gnp = accuracy_score(y_test, y_predict2)

print(accuracy_gnp, accuracy_svm)
print(svm.predict([HuMoments(PreProcess("13.png")).flatten()]))
