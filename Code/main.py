import cv2
from Segmentation.Segmentor import Segmentor


if __name__ == "__main__":
    img = cv2.imread("capr4.png")
    Segmentor(img).run_segmentor()
    cv2.waitKey(0)
    