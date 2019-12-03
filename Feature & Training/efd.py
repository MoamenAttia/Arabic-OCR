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
    cv2.imwrite('a.png', im)
    cv2.imshow("ss", im)
    contours, hierarchy = cv2.findContours(
        im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    g = cv2.drawContours(im, contours, -1, (0, 255, 0), 3)

    '''
    coeffs = []
    for cnt in contours:
        efd_feature(cnt)
        break
    '''


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


im = PreProcess("2.png")

print(efd(im))
cv2.waitKey(0)
