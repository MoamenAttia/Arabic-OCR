import cv2
import numpy as np
import os
import glob
import re
from Line_segmentation import segment_paragragh
from WordSegmentor import WordSegmentor
from LineSegmentor import LineSegmentor


input_path = '../input'

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def skew_correction(image):
    gray = cv2.bitwise_not(image)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

input_imgs = []
for filename in sorted(glob.glob(os.path.join(input_path, '*.png')), key=numericalSort):
    print(filename)
    _imgs = cv2.imread(filename, 0)
    input_imgs.append(_imgs)

for i in range(len(input_imgs)):
    img = skew_correction(input_imgs[i])
    lines, lines_dil = LineSegmentor(img).segment_lines()
    words, length = WordSegmentor(lines, lines_dil).segment_words()  
    lt_img = segment_paragragh(lines, words) #array of arrays eac array contains chars imgs of word
    cv2.imwrite('test.png', lt_img[0][0])
    # cont.