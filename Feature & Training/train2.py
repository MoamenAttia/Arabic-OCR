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
import time


def efd_feature(contour):
    coeffs = elliptic_fourier_descriptors(
        contour, order=10, normalize=True)
    return coeffs.flatten()[3:]


def efd(binary):
    contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]

    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    features = efd_feature(np.array(contours[1]).transpose(
        2, 0, 1).reshape(np.array(contours[1]).shape[0], 2))

    return features


def dct(img):
    imf = np.float32(img)/255.0  # float conversion/scale
    dcts = cv2.dct(imf)           # the dct
    return dcts


def HuMoments(img):
    moments = cv2.moments(img)
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)

    if [0] in huMoments:
        return np.zeros(7)
    for i in range(0, 7):
        huMoments[i] = -1 * \
            copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
    return huMoments


def PreProcess(im, option):
    img = cv2.imread(im)
    if option == 0:
        return_img = img
    if option == 1:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return_img = gray
    elif option == 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        return_img = binary

    return return_img


folders = ['alif', 'baa', 'taa', 'seh', 'jiim', 'haa', 'khaa', 'daal']
dataset_binary = []
dataset_gray = []
labels = np.zeros(8000)
labels[1000:1999] = 1
labels[2000:2999] = 2
labels[3000:3999] = 3
labels[4000:4999] = 4
labels[5000:5999] = 5
labels[6000:6999] = 6
labels[7000:7999] = 7


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
        if i > 1000:
            break
        gray = PreProcess(elem, 1)
        binary = PreProcess(elem, 2)
        dataset_binary.append(binary)
        dataset_gray.append(gray)


feature_vector = []
for i in range(len(dataset_binary)):
    # features_1 = efd(dataset_binary[i])
    # features_2 = HuMoments(dataset_binary[i]).flatten()
    features_3 = dct(dataset_gray[i]).flatten()

    feature_vector.append(np.array(features_3))
    label.append(labels[i])


scaler = StandardScaler()
scaler.fit(feature_vector)
scaler.transform(feature_vector)

feature_matrix = np.array(feature_vector)

x = pd.DataFrame(feature_matrix)
y = pd.Series(label)


print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

svm = SVC(kernel="linear")

millis = int(round(time.time() * 1000))
svm.fit(x_train, y_train)
y_predict = svm.predict(x_test)
millis2 = int(round(time.time() * 1000))
millis_svm = millis2-millis


gnp = GaussianNB()

millis = int(round(time.time() * 1000))
gnp.fit(x_train, y_train)
y_predict2 = gnp.predict(x_test)
millis2 = int(round(time.time() * 1000))
millis_gnb = millis2-millis


accuracy_svm = accuracy_score(y_test, y_predict)
accuracy_gnp = accuracy_score(y_test, y_predict2)

print(accuracy_gnp, accuracy_svm)
print(millis_gnb, millis_svm)

cv2.waitKey(0)
