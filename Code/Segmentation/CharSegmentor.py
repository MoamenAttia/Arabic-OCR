import cv2
import numpy as np


class CharSegmentor:
    def __init__(self, line_words, threshold=0):
        self.__line_words = line_words
        self.__threshold = threshold

    def __detect_baseline(self, line):
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(line, kernel, iterations=1)
        kernel = np.ones((3, 3), np.float32)/9
        line = cv2.filter2D(erosion, -1, kernel)
        hist = np.sum(255 - line, axis=1)
        max_hist = 0
        for i in range(len(hist)):
            if hist[i] >= max_hist:
                max_hist = hist[i]
                baseline = i
        return baseline

    def __detect_maximum_transition_line(self, line, baseline):
        max_transitions = 0
        max_transitions_index = baseline
        i = baseline
        temp = 255 - line
        _, thresh = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
        for i in range(0, baseline):
            curr_transitions = 0
            flag = 0
            for j in range(len(thresh[0])):
                if thresh[i, j] == 255 and flag == 0:
                    curr_transitions += 1
                    flag = 1
                elif thresh[i, j] == 0 and flag == 1:
                    flag = 0
            if curr_transitions >= max_transitions:
                max_transitions = curr_transitions
                max_transitions_index = i
        return max_transitions_index

    def segment_chars(self):
        for line, words in self.__line_words:
            baseline = self.__detect_baseline(line)
            max_transition_line = self.__detect_maximum_transition_line(line, baseline)