import cv2
import numpy as np
from scipy import stats
import skimage.graph


class CharSegmentor:
    def __init__(self, line_words, threshold=0):
        self.__line_words = line_words
        self.__threshold = threshold

    def __detect_baseline(self, line):
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(line, kernel, iterations=1)
        kernel = np.ones((3, 3), np.float32) / 9
        line = cv2.filter2D(erosion, -1, kernel)
        hist = np.sum(255 - line, axis=1)
        [max_hist, baseline] = [0, 0]
        for i in range(len(hist)):
            if hist[i] >= max_hist:
                max_hist = hist[i]
                baseline = i
        return baseline

    def __comp(self, x, y, mid, s, f, cond):
        ress = []
        if cond == '==':
            for i in range(f, s):
                if x[i] == y:
                    ress.append(i)
        else:
            for i in range(f, s):
                if x[i] <= y:
                    ress.append(i)
        if len(ress) != 0:
            ress = np.asarray(ress)
            return ress.flat[np.abs(ress - mid).argmin()]
        else:
            return False

    def __is_connected_component(self, word, start, end):
        return 0 not in np.sum(255 - word[::, start:end + 1], axis=0)

    def __cutting_points_identification(self, line, word, mti):
        vp = np.sum(255 - word, axis=0)
        i = 0
        flag = 0
        m = stats.mode(vp)
        mfv = m[0][0]
        p = [value for value in vp if value != 0]
        n = stats.mode(p)
        mfv1 = n[0][0]
        [StartIndex, EndIndex, MidIndex, CutIndex] = [0, 0, 0, 0]
        sr = []
        while i < word.shape[1]:
            if word[mti, i] == 0 and flag == 0:
                EndIndex = i
                flag = 1
            elif word[mti, i] != 0 and flag == 1:
                StartIndex = i
                MidIndex = int((EndIndex + StartIndex) / 2)
                if self.__comp(vp, 0, MidIndex, StartIndex, EndIndex, '=='):
                    CutIndex = self.__comp(vp, 0, MidIndex, StartIndex, EndIndex, '==')
                elif vp[MidIndex] == mfv:
                    CutIndex = MidIndex
                elif self.__comp(vp, mfv, MidIndex, MidIndex, EndIndex, '<='):
                    CutIndex = self.__comp(vp, mfv, MidIndex, MidIndex, EndIndex, '<=')
                elif self.__comp(vp, mfv, MidIndex, StartIndex, MidIndex, '<='):
                    CutIndex = self.__comp(vp, mfv, MidIndex, StartIndex, MidIndex, '<=')
                else:
                    CutIndex = MidIndex
                sr.append([StartIndex, EndIndex, MidIndex, CutIndex])
                flag = 0
            i += 1

        return mfv1, mfv, sr

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

    def __does_hole_exist(self, word, max_transition_idx, start, end):
        return 0 in np.sum(255 - word[::, start:end + 1], axis=0)

    def __does_baseline_exist(self, word, baseline_idx, start, end):
        for i in range(start, end):
            if word[baseline_idx, i] == 0:
                return False
        return True

    def __calculate_height(self, region):
        height = 0
        for i in range(len(region)):
            height = 0
            for j in range(len(region[0])):
                if region[j, i] == 0:
                    height = max(height, len(region) - i)
                    break
        return height

    def __is_segment_height_less_than_half_line_height(self, line, word, start, end):
        line_height = self.__calculate_height(line)
        segment_height = self.__calculate_height(word[::, start:end + 1])
        return segment_height <= line_height

    def __does_dots_exist(self, word, start, end):
        segment = word[::, start:end + 1]
        hp = np.sum(255 - word, axis=1)
        [count, flag] = [0, 0]
        for j in range(len(hp)):
            if hp[j] > 0 and flag == 0:
                count += 1
                flag = 1
            elif hp[j] == 0 and flag == 1:
                flag = 0
        return count > 1

    def __is_stroke(self, word, baseline_index, max_transition_index, mfv, start, end):
        # single connected component,
        # the sum of horizontal projection above baseline is greater than the sum of horizontal projection below baseline,
        # the height of the segment is less than twice the second peak value of the horizontal projection,
        # the mode value of the horizontal projection is equal to MFV value,
        # the segment has no Holes.
        hp = np.sum(255 - word, axis=1)
        shpa = np.sum(hp[0:baseline_index], axis=0)
        shpb = np.sum(hp[baseline_index + 1:], axis=0)
        segment_height = self.__calculate_height(word[::, start:end + 1])
        second_peak = list(sorted(hp, reverse=True))[1] if len(hp) > 1 else 0
        return hp[stats.mode(hp)[0][0]] == mfv and self.__is_connected_component(word, start,
                                                                                 end) and shpa > shpb and segment_height < 2 * second_peak and self.__does_hole_exist(
            word, max_transition_index, start, end)

    def __filter_separation_region(self, line, word, sr_list, baseline_index, max_transitions_index, mfv):
        i = 0
        valid_sr = []
        while i < len(sr_list):
            sr = sr_list[i]
            vp = np.sum(255 - line, axis=0)
            prev_cut = sr_list[i - 1][3] if i - 1 >= 0 else sr_list[i][3]
            next_cut = sr_list[i + 1][3] if i + 1 < len(sr_list) else sr_list[i][3]

            if vp[sr[3]] == 0:
                valid_sr.append(sr)
                i += 1
            elif not self.__is_connected_component(word, sr[0], sr[1]):
                valid_sr.append(sr)
                i += 1
            elif self.__does_hole_exist(word, max_transitions_index, prev_cut, next_cut):
                i += 1
            elif self.__does_baseline_exist(word, baseline_index, sr[0], sr[1]):
                hp = np.sum(255 - word, axis=1)
                shpa = np.sum(hp[0:baseline_index], axis=0)
                shpb = np.sum(hp[baseline_index + 1:], axis=0)
                if shpb > shpa:
                    i += 1
                elif vp[sr[3]] < mfv:
                    valid_sr.append(sr)
                    i += 1
                else:
                    i += 1
            elif i == len(sr_list) - 1 or sr_list[i + 1][
                3] == 0 and self.__is_segment_height_less_than_half_line_height(line, word, sr[0], sr[1]):
                i += 1
            elif not self.__is_stroke(word, baseline_index, max_transitions_index, mfv, sr[3], sr_list[i + 1][3]):
                if not self.__does_baseline_exist(word, baseline_index, sr_list[i + 1][0], sr_list[i + 1][1]) and \
                        sr_list[i + 1][3] <= stats.mode(vp)[0][0]:
                    i += 1
                else:
                    valid_sr.append(sr)
                    i += 1
            elif self.__is_stroke(word, baseline_index, max_transitions_index, mfv, sr[3],
                                  sr_list[i + 1][3]) and self.__does_dots_exist(word, sr[3], sr_list[i + 1][3]):
                valid_sr.append(sr)
                i += 1
            elif self.__is_stroke(word, baseline_index, max_transitions_index, mfv, sr[3],
                                  sr_list[i + 1][3]) and not self.__does_dots_exist(word, sr[3], sr_list[i + 1][3]):
                if self.__is_stroke(word, baseline_index, max_transitions_index, mfv, sr_list[i + 1][3],
                                    sr_list[i + 2][3]) and not self.__does_dots_exist(word, sr_list[i + 1][3],
                                                                                      sr_list[i + 2][3]):
                    valid_sr.append(sr)
                    i += 3
                if self.__is_stroke(word, baseline_index, max_transitions_index, mfv, sr_list[i + 1][3],
                                    sr_list[i + 2][3]) and self.__does_dots_exist(word, sr_list[i + 1][3],
                                                                                  sr_list[i + 2][
                                                                                      3]) and self.__is_stroke(word,
                                                                                                               baseline_index,
                                                                                                               max_transitions_index,
                                                                                                               mfv,
                                                                                                               sr_list[
                                                                                                                   i + 2][
                                                                                                                   3],
                                                                                                               sr_list[
                                                                                                                   i + 3][
                                                                                                                   3]) and not self.__does_dots_exist(
                        word, sr_list[i + 2][3], sr_list[i + 3][3]):
                    valid_sr.append(sr)
                    i += 3
                if not self.__is_stroke(word, baseline_index, max_transitions_index, mfv, sr_list[i + 1][3],
                                        sr_list[i + 2][3]) or (
                        self.__is_stroke(word, baseline_index, max_transitions_index, mfv, sr_list[i + 1][3],
                                         sr_list[i + 2][3]) and self.__does_dots_exist(word, sr_list[i + 1][3],
                                                                                       sr_list[i + 2][3])):
                    i += 1
        return valid_sr

    def segment_chars(self):
        for line, words in self.__line_words:
            baseline = self.__detect_baseline(line)
            max_transition_index = self.__detect_maximum_transition_line(line, baseline)
            for word in words:
                [mfv1, mfv, sr_list] = self.__cutting_points_identification(line, word, max_transition_index)
                filtered_regions = self.__filter_separation_region(line, word, sr_list, baseline, max_transition_index, mfv)
                for i in range(len(filtered_regions)):
                    word[0:word.shape[0], filtered_regions[i][3]] = 0
                cv2.imwrite('zoo.png', word)