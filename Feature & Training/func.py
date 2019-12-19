import cv2
import numpy as np


i = cv2.imread('capr6.png', 0)
ret, img = cv2.threshold(i, 127, 255, cv2.THRESH_BINARY_INV)
size = np.size(img)
skel = np.zeros(img.shape, np.uint8)

ret, img = cv2.threshold(img, 127, 255, 0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
done = False

while(not done):
    eroded = cv2.erode(img, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(img, temp)
    skel = cv2.bitwise_or(skel, temp)
    img = eroded.copy()

    zeros = size - cv2.countNonZero(img)
    if zeros == size:
        done = True

cv2.imshow("skel", skel)
i = cv2.imread('capr6.png', 0)
ret, img = cv2.threshold(i, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("ske-l", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
