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
    #return_img = np.pad(return_img,((1,1),(1,1)),mode='constant',constant_values=255)
    return return_img


'''
folders = ['alif', 'baa', 'taa', 'seh', 'jiim', 'haa', 'khaa', 'daal',  'raa', 'siin', 'shiin',
           'saad', 'daad', 'tah', 'een', 'ghin', 'faa', 'qaaf', 'kaaf', 'laam', 'miim', 'noon', 'heh', 'waaw',
           'laamalif' ,'zaal', 'zeen', 'zaa', 'yaa2']
'''
folders = ['alif', 'baa', 'taa', 'seh', 'jiim',
           'haa', 'khaa', 'daal',  'raa', 'siin', 'shiin', 'saad', 'daad', 'tah', 'een', 'ghin', 'faa', 'qaaf',
           'kaaf', 'laam', 'miim', 'noon', 'heh', 'waaw', 'laamalif', 'zaal', 'zeen', 'zaa', 'yaa2'
           ]

dataset = []
labels = np.zeros(len(folders)*1000)
labels[1000:1999] = 1
labels[2000:2999] = 2
labels[3000:3999] = 3
labels[4000:4999] = 4
labels[5000:5999] = 5
labels[6000:6999] = 6
labels[7000:7999] = 7
labels[8000:8999] = 8
labels[9000:9999] = 9
labels[10000:10999] = 10
labels[11000:11999] = 11
labels[12000:1299] = 12
labels[13000:13999] = 13
labels[14000:14999] = 14
labels[15000:15999] = 15
labels[16000:16999] = 16
labels[17000:17999] = 17

labels[18000:18999] = 18
labels[19000:19999] = 19

labels[20000:20999] = 20

labels[21000:21999] = 21
labels[22000:22999] = 22

labels[23000:23999] = 23
labels[24000:24999] = 24

labels[25000:25999] = 25

labels[26000:26999] = 26
labels[27000:27999] = 27

labels[28000:28999] = 28
labels[29000:29999] = 29


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
        if i > 999:
            break
        gray = PreProcess(elem, 1)
        binary = PreProcess(elem, 2)
        dataset.append(gray)


feature_vector = []
for i in range(len(dataset)):
    feature_vector.append(dct(dataset[i]).flatten())


scaler = StandardScaler()
scaler.fit(feature_vector)
scaler.transform(feature_vector)

feature_matrix = np.array(feature_vector)

x = pd.DataFrame(feature_matrix)
y = pd.Series(labels)


print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

svm = SVC(kernel="rbf")
svm.fit(x_train, y_train)

gnp = GaussianNB()
gnp.fit(x_train, y_train)

y_predict = svm.predict(x_test)
y_predict2 = gnp.predict(x_test)

accuracy_svm = accuracy_score(y_test, y_predict)
accuracy_gnp = accuracy_score(y_test, y_predict2)

print(accuracy_gnp, accuracy_svm)

img = PreProcess('13.png', 1)
features = dct(img).flatten()
print(gnp.predict([features]), svm.predict([features]))
