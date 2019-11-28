import cv2
import numpy as np

from .LineSegmentor import LineSegmentor
from .WordSegmentor import WordSegmentor
from .CharSegmentor import CharSegmentor

class Segmentor:
    def __init__(self, img):
        self.__img = img
        self.__gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    def __print_lines(self, lines):
        for idx, line in enumerate(lines):
            cv2.imwrite(f'lines/line{idx}.png', line)

    def run_segmentor(self):
        lines = LineSegmentor(self.__gray_image).segment_lines()
        self.__print_lines(lines)
        line_words = WordSegmentor(lines).segment_words()
        CharSegmentor(line_words).segment_chars()
