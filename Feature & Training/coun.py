import cv2
import numpy as np
import imutils

img = cv2.imread("n.png")
cv2.imshow("Original image", img)

grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray image", grayImg)

ret, binImg = cv2.threshold(grayImg, 100, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary image", binImg)

contours = cv2.findContours(binImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if imutils.is_cv2() else contours[1]

print("Contours number found:", len(contours))
iContours = 0

for contour in contours:
    cv2.drawContours(img, contour, iContours, (0, 255, 0))
    iContours += 1

cv2.imshow("Original image", img)

cv2.waitKey(0)
