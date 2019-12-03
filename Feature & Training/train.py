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
    cv2.imshow("ss", img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return binary


folders = ['wow', 'kaf', 'lam', 'seen', 'dal', 'ha2', 'kaf', 'ba2']
#folders = ['wow', 'kaf', 'lam']
dataset = []
labels = np.zeros(2400)
labels[300:599] = 1
labels[600:899] = 2
labels[900:1199] = 3
labels[1200:1499] = 4
labels[1500:1799] = 5
labels[1800:2099] = 6
labels[2100:2399] = 7

for folder in folders:
    dirName = os.path.join("C:/Users\medo\Desktop\pattern\dataset_sep", folder)
    print(dirName)

    listOfFiles = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(dirName):
        for file in f:

            listOfFiles.append(os.path.join(r, file))

    # Print the files
    for i, elem in enumerate(listOfFiles):

        binary = PreProcess(elem)
        dataset.append(binary)
        if i > 298:
            break


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
print(svm.predict([HuMoments(PreProcess("2.png")).flatten()]))
cv2.waitKey(0)
