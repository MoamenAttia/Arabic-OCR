import cv2
import numpy as np
import os
import glob
import re
import time
from Line_segmentation import segment_paragragh
from WordSegmentor import WordSegmentor
from LineSegmentor import LineSegmentor
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
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
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from numpy import array
from numpy import argmax
from sklearn.preprocessing import OneHotEncoder
from keras.models import model_from_json
from tempfile import TemporaryFile
from padding import pad


input_path = 'C:/Users\medo\Desktop\Arabic-OCR\Code\input'
output_path = 'C:/Users\medo\Desktop\Arabic-OCR\Code\output\\'

numbers = re.compile(r'(\d+)')


def dct(img):
    imf = np.float32(img)/255.0  # float conversion/scale
    dcts = cv2.dct(imf)           # the dct
    return dcts


def PreProcess(im, option):

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


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def skew_correction(image):
    gray = cv2.bitwise_not(image)

    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


final_labels = {'alif': 'ا', 'baa': 'ب', 'taa': 'ت',
                'seh': 'ث',
                'jiim': 'ج',
                'haa': 'ح',
                'khaa': 'خ',
                'daal': 'د',
                'raa': 'ر',
                'siin': 'س',
                'shiin': 'ش',
                'saad': 'ص',
                'daad': 'ض',
                'tah': 'ط',
                'een': 'ع',
                'ghin': 'غ',
                'faa': 'ف',
                'qaaf': 'ق',
                'kaaf': 'ك',
                'laam': 'ل',
                'miim': 'م',
                'noon': 'ن',
                'heh': 'ه',
                'waaw': 'و',
                'laamalif': 'لا',
                'zaal': 'ذ',
                'zeen': 'ز',
                'zaa': 'ظ',
                'yaa2': 'ي'
                }


# load labels
Y = np.genfromtxt(
    'C:/Users\medo\Desktop\Arabic-OCR\Code\code\labels.txt', dtype='str')
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

# load model
json_file = open(
    'C:/Users\medo\Desktop\Arabic-OCR\Code\code\model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(
    "C:/Users\medo\Desktop\Arabic-OCR\Code\code\model.h5")
print("Loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy',
                     optimizer='adam', metrics=['accuracy'])

input_imgs = []
for filename in sorted(glob.glob(os.path.join(input_path, '*.png')), key=numericalSort):
    print(filename)
    _imgs = cv2.imread(filename, 0)
    input_imgs.append(_imgs)

directory = output_path+'running_time.txt'

out_time = open(directory, 'w+', encoding='utf-8')


# time here
for i in range(len(input_imgs)):
    out = open('C:/Users\medo\Desktop\Arabic-OCR\Code\output\\text\\test_' + str(i+1)+'.txt',
               'w+', encoding='utf-8')
    final_output = []
    seconds1 = time.time()
    img = skew_correction(input_imgs[i])
    lines, lines_dil = LineSegmentor(img).segment_lines()
    words, length = WordSegmentor(lines, lines_dil).segment_words()
    # array of arrays each array contains chars imgs of word
    lt_img = segment_paragragh(lines, words)
    for i, word in enumerate(lt_img):
        for j, letter in enumerate(word):
            letter = pad(letter)
            features = np.array(dct(letter).flatten()).reshape(1, 2500)
            res = loaded_model.predict(features)
            inverted = encoder.inverse_transform([argmax(res[0])])
            final_output.append(final_labels[inverted[0]])
        final_output.append(' ')
    seconds2 = time.time()
    final_time = seconds2-seconds1
    out_time.write("%s\n" % final_time)
    for char in final_output:
        out.write(char)
