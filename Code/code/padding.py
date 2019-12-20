import cv2
import numpy as np
import glob
import os
import re


def pad(img):
    if img.shape[0] > 50:
        img = cv2.resize(img, (img.shape[1], 50), interpolation=cv2.INTER_AREA)
        # return img
    if img.shape[1] > 50:
        img = cv2.resize(img, (50, img.shape[0]), interpolation=cv2.INTER_AREA)

    height_needed = 50 - img.shape[0]
    width_needed = 50 - img.shape[1]

    top = int(np.floor(height_needed / 2))
    bottom = int(np.ceil(height_needed / 2))
    left = int(np.floor(width_needed / 2))
    right = int(np.ceil(width_needed / 2))

    border = cv2.copyMakeBorder(
        img,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=255
    )
    return border


img = "C:/Users\medo\Desktop\pattern2\dataset\\"


folders = ['laamalif']

# folders = ['laam']
numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


for f in folders:
    imgs = []
    for filename in sorted(glob.glob(os.path.join(img + f, '*.png')), key=numericalSort):
        #    print(filename)
        _imgs = cv2.imread(filename, 0)
        imgs.append(_imgs)

    temp = []
    for x in imgs:
        temp.append(pad(x))

    i = 0
    os.chdir(img + f)
    for x in temp:
        cv2.imwrite(str(i)+'.png', x)
        i += 1

#imgs = []
# for filename in sorted(glob.glob(os.path.join(img, '*.png')), key=numericalSort):
#    print(filename)
#    _imgs = cv2.imread(filename, 0)
#    imgs.append(_imgs)

#temp = []
# for x in imgs:
#    temp.append(pad(x))

# os.chdir(dst)
#i = 0
# for x in temp:
#    cv2.imwrite(str(i)+'.png', x)
#    i += 1
