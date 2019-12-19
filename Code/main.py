import cv2
from Segmentation.Segmentor import Segmentor
import numpy as np


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


if __name__ == "__main__":
    img = cv2.imread("csep600.png", 0)
    img = skew_correction(img)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    Segmentor(img).run_segmentor()
    cv2.waitKey(0)
