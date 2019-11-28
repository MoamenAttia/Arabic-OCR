import cv2
import numpy as np


class LineSegmentor:
    def __init__(self, img, threshold=0):
        """
        Takes an image of the line.
            :param img: array representing the image [ gray image ].
            :param threshold: int [default = 0] => may change threshlod in case of inclined lines.
        """
        self.__img = img
        self.__threshold = threshold

    def segment_lines(self):
        hist = np.sum(255 - self.__img, axis=1)
        i = 0
        lines = []
        while i < len(hist):
            if hist[i] != 0:
                x, y = i, i
                while y < len(hist):
                    y += 1
                    if y < len(hist) and hist[y] == 0:
                        break
                lines.append(self.__img[x:y, 0:len(self.__img[0])])
                i = y
            else:
                i += 1
        return lines
