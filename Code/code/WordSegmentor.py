import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)


class WordSegmentor:
    def __init__(self, lines, lines_dil, threshold=2):
        self.__lines = lines
        self.__lines_dil = lines_dil

        # to identify how many pixels to be declared as mid-gap ( not space )
        self.__threshold = threshold

    def __find_gaps(self, line):
        hist = np.sum(255 - line, axis=0)
        gaps = []
        i = 0
        while i < len(hist):
            if hist[i] == 0:
                y = i + 1
                while True:
                    if y < len(hist) and hist[y] == 0:
                        y += 1
                    else:
                        if y - i > self.__threshold:
                            gaps.append([i, y - 1])
                        break
                i = y
                continue
            i += 1
        return gaps

    def __segment_line(self, line, line_dil):
        gaps = self.__find_gaps(line_dil)
        words = []
        for i in range(0, len(gaps) - 1):
            words.append(line[::, gaps[i][1]:gaps[i+1][0]])
        if len(line[0]) - 1 - gaps[-1][1] > 0:
            words.append(line[::, gaps[-1][1]:])
        if gaps[0][0] != 0:
            words.insert(0, line[::, 0:gaps[0][0]])
        words.reverse()
        return words

    def segment_words(self):
        line_words = []
        length = 0
        for line_idx, line in enumerate(self.__lines):
            #print(len(self.__lines), len(self.__lines_dil))
            words = self.__segment_line(line, self.__lines_dil[line_idx])
            length += len(words)
            line_words.append(words)

            #for idx, word in enumerate(words):
            #    cv2.imwrite(f'line{line_idx}-word{idx}.png', word)
            #cv2.imwrite(f'line{line_idx}.png', line)

        return line_words, length

