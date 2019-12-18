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

    def __print_words(self, line_words):
        count = 0
        for item in line_words:
            for _, word in enumerate(item[1]):
                cv2.imwrite(f'words/word{count}.png', word)
                count += 1


    def run_segmentor(self):
        lines = LineSegmentor(self.__gray_image).segment_lines()
        line_words = WordSegmentor(lines).segment_words()
        print(f"lines count {len(line_words)}")

        self.__print_words(line_words)
        # CharSegmentor(line_words).segment_chars()
