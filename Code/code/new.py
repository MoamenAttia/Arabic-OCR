import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy import stats
import skimage.graph

def __path_cost(skeleton, img, mti, s, t):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 255:
                skeleton[i, j] = True
            else:
                skeleton[i, j] = False
    array = np.asarray(skeleton)
    costs = np.where(array, 0.1, 10e20)
    path, cost = skimage.graph.route_through_array(costs, start=(mti, s - 1), end=(mti, t), fully_connected=True)

    return cost

def __horizintal_projection(im):
    projection = np.sum(im, 1)  
    return projection


def __vertical_projection(im):
    projection = np.sum(im, 0)
    return projection

def __does_hole_exist(word, boundaries, mti):
    img = word[0:mti, 0:word.shape[1]]
    vp_img = __vertical_projection(img)

    for i in range(boundaries[1], boundaries[0]):
        if vp_img[i] == 0:
            return False
    return True

def __does_baseline_exist(word, boundaries, bl):
    for i in range(boundaries[1], boundaries[3]):
        if word[bl, i] == 0:
            return False
  
    return True

def get_SHP(word, boundaries, start, end):
    img = word[start:end, boundaries[1]:boundaries[0]]
    hp_img = __horizintal_projection(img)
    return np.sum(hp_img)


def __highest_left_pixel(word, boundaries1, boundaries2):
    img = word[0:word.shape[0], boundaries2[3]:boundaries1[3]]
    vp_img = __vertical_projection(img)

    i = 0
    while vp_img[i] == 0:
        i += 1

    j = 0
    while j < word.shape[0]:
        if img[j, i] != 0:
            return j
        j += 1

    return word.shape[0]


def __is_connected_component(img, bl):
    count = 0
    flag = 0
    for j in range(img.shape[1]):
        if img[bl, j] == 255 and flag == 0:
            count += 1
            flag = 1
        elif img[bl, j] != 255 and flag == 1:
            flag = 0

    return count

def __is_stroke(word, boundaries1, boundaries2, bl, top_pixel_in_word_index, mfv, mfv1, mti, second, skeleton, vp):
    img = word[0:word.shape[0], boundaries1[3]:boundaries2[3]]
    start, end = __cropping_indeces(__horizintal_projection(img), bl)
    img1 = word[start:end, boundaries1[3]:boundaries2[3]]

    if __is_connected_component(img, int(word.shape[0]/2)) > 1:
        return False

    if get_SHP(img1, [img1.shape[1], 0], 0, bl) <= get_SHP(img1, [img1.shape[1], 0], bl, img1.shape[0]):
        return False

    hp_img = __horizintal_projection(img1)
    height_of_seg = bl - start

    second_peak = bl - second
    if height_of_seg > second_peak or start <= top_pixel_in_word_index + 2:

        return False

    hp_img = __horizintal_projection(img1)
    hp_img = [value for value in hp_img if value != 0]
    m = stats.mode(hp_img)
    mode_hp = m[0][0]
    if mfv == 0:
        mfv = mfv1
    if mode_hp not in range(mfv - 600, mfv + 600):

        return False

    # cond5:
    if (__does_hole_exist(img, [img.shape[1], 0], mti) and not __does_dots_exist(word, boundaries1, boundaries2)) or __does_hole_exist(word, boundaries1, mti) or __path_cost(skeleton, word, mti, boundaries1[1], boundaries1[0]) > 10e20 or vp[boundaries1[3]] == 0:
        return False
    return True

def __does_dots_exist(word, boundaries1, boundaries2):
    img = word[0:word.shape[0], boundaries1[3] + 2:boundaries2[3] - 2]
    hp_img = __horizintal_projection(img)
    count = 0
    flag = 0
    for j in range(len(hp_img)):
        if hp_img[j] > 0 and flag == 0:
            count += 1
            flag = 1
        elif hp_img[j] == 0 and flag == 1:
            flag = 0

    if count > 1:
        return True
    return False

def filter_cutting_points(skeleton, word, sr, baseLine, maxTransitionIndex, most_frequent_value, most_frequent_value_after_0, vp, hp, second):
    i = 0
    sr.append([word.shape[1] - 1, word.shape[1] - 1,
               word.shape[1] - 1, word.shape[1] - 1])
    valid_sr = []
    top_pixel_in_word_index = __heighest_pixel_index(hp)
    sr.reverse()

    if i < len(sr) - 1 and __is_stroke(word, sr[i + 1], sr[i], baseLine, top_pixel_in_word_index, most_frequent_value,
                                    most_frequent_value_after_0, maxTransitionIndex, second, skeleton,
                                    vp) and not __does_dots_exist(word, sr[i + 1], sr[i]):
        if i < len(sr) - 2 and __is_stroke(word, sr[i + 2], sr[i + 1], baseLine, top_pixel_in_word_index,
                                        most_frequent_value,
                                        most_frequent_value_after_0, maxTransitionIndex, second, skeleton,
                                        vp) and not __does_dots_exist(word, sr[i + 2], sr[i + 1]):
            valid_sr.append(sr[i])
            i += 3
        elif i < len(sr) - 3 and __is_stroke(word, sr[i + 2], sr[i + 1], baseLine, top_pixel_in_word_index,
                                          most_frequent_value,
                                          most_frequent_value_after_0, maxTransitionIndex, second, skeleton,
                                          vp) and __does_dots_exist(word, sr[i + 2], sr[i + 1]) and __is_stroke(word, sr[i + 3],
                                                                                                     sr[i + 2],
                                                                                                     baseLine,
                                                                                                     top_pixel_in_word_index,
                                                                                                     most_frequent_value,
                                                                                                     most_frequent_value_after_0,
                                                                                                     maxTransitionIndex,
                                                                                                     second, skeleton,
                                                                                                     vp) and not __does_dots_exist(
                word, sr[i + 3], sr[i + 2]):
            valid_sr.append(sr[i])
            i += 3

        else:
            i += 1


    valid_sr.append(sr[0])
    while i < len(sr) - 1:
        cost = __path_cost(
            skeleton, word, maxTransitionIndex, sr[i][1], sr[i][0])

        if vp[sr[i][3]] == 0:
            if i < len(sr) - 1 and __is_stroke(word, sr[i + 1], sr[i], baseLine, top_pixel_in_word_index,
                                            most_frequent_value,
                                            most_frequent_value_after_0, maxTransitionIndex, second, skeleton,
                                            vp) and not __does_dots_exist(word, sr[i + 1], sr[i]):
                if i < len(sr) - 2 and __is_stroke(word, sr[i + 2], sr[i + 1], baseLine, top_pixel_in_word_index,
                                                most_frequent_value,
                                                most_frequent_value_after_0, maxTransitionIndex, second, skeleton,
                                                vp) and not __does_dots_exist(word, sr[i + 2], sr[i + 1]):
                    valid_sr.append(sr[i])
                    i += 3
                elif i < len(sr) - 3 and __is_stroke(word, sr[i + 2], sr[i + 1], baseLine, top_pixel_in_word_index,
                                                  most_frequent_value,
                                                  most_frequent_value_after_0, maxTransitionIndex, second, skeleton,
                                                  vp) and __does_dots_exist(word, sr[i + 2], sr[i + 1]) and __is_stroke(word,
                                                                                                             sr[i + 3],
                                                                                                             sr[i + 2],
                                                                                                             baseLine,
                                                                                                             top_pixel_in_word_index,
                                                                                                             most_frequent_value,
                                                                                                             most_frequent_value_after_0,
                                                                                                             maxTransitionIndex,
                                                                                                             second,
                                                                                                             skeleton,
                                                                                                             vp) and not __does_dots_exist(
                        word, sr[i + 3], sr[i + 2]):
                    valid_sr.append(sr[i])
                    i += 3
                elif __heighest_pixel_index(__horizintal_projection(word[0:word.shape[0], sr[i+1][3]:sr[i][3]])) < maxTransitionIndex - 5:
                    valid_sr.append(sr[i])
                    i += 1
                else:
                    i += 1
            else:
                valid_sr.append(sr[i])
                i += 1

        elif cost > 10e20:
            if i < len(sr) - 1 and __is_stroke(word, sr[i + 1], sr[i], baseLine, top_pixel_in_word_index,
                                            most_frequent_value,
                                            most_frequent_value_after_0, maxTransitionIndex, second, skeleton,
                                            vp) and not __does_dots_exist(word, sr[i + 1], sr[i]):
                if i < len(sr) - 2 and __is_stroke(word, sr[i + 2], sr[i + 1], baseLine, top_pixel_in_word_index,
                                                most_frequent_value,
                                                most_frequent_value_after_0, maxTransitionIndex, second, skeleton,
                                                vp) and not __does_dots_exist(word, sr[i + 2], sr[i + 1]):
                    valid_sr.append(sr[i])
                    i += 3
                elif i < len(sr) - 3 and __is_stroke(word, sr[i + 2], sr[i + 1], baseLine, top_pixel_in_word_index,
                                                  most_frequent_value,
                                                  most_frequent_value_after_0, maxTransitionIndex, second, skeleton,
                                                  vp) and __does_dots_exist(word, sr[i + 2], sr[i + 1]) and __is_stroke(word,
                                                                                                             sr[i + 3],
                                                                                                             sr[i + 2],
                                                                                                             baseLine,
                                                                                                             top_pixel_in_word_index,
                                                                                                             most_frequent_value,
                                                                                                             most_frequent_value_after_0,
                                                                                                             maxTransitionIndex,
                                                                                                             second,
                                                                                                             skeleton,
                                                                                                             vp) and not __does_dots_exist(
                        word, sr[i + 3], sr[i + 2]):
                    valid_sr.append(sr[i])
                    i += 3
                elif __heighest_pixel_index(__horizintal_projection(word[0:word.shape[0], sr[i+1][3]:sr[i][3]])) < maxTransitionIndex - 5:
                    valid_sr.append(sr[i])
                    i += 1

                else:

                    i += 1
            else:
                valid_sr.append(sr[i])
                i += 1

        elif __does_hole_exist(word, sr[i], maxTransitionIndex):
            i += 1

        elif not __does_baseline_exist(word, sr[i], baseLine):
            if get_SHP(word, sr[i], baseLine, word.shape[0]) > get_SHP(word, sr[i], 0, baseLine):
                i += 1

            elif vp[sr[i][3]]/255 < most_frequent_value/255:
                valid_sr.append(sr[i])
                i += 1
            else:
                i += 1

        # might need to change operator
        elif (vp[sr[i + 1][3]] == 0 or i == len(sr) - 2) and (-__highest_left_pixel(word, sr[i], sr[i + 1]) + baseLine) < int((-top_pixel_in_word_index + baseLine)/2)\
                and __path_cost(skeleton, word, maxTransitionIndex, sr[i][1], sr[i][0]) < 1000000000 and vp[sr[i+1][3]] != 0:
            i += 1

        elif i < len(sr) - 1 and not __is_stroke(word, sr[i + 1], sr[i], baseLine, top_pixel_in_word_index, most_frequent_value, most_frequent_value_after_0, maxTransitionIndex, second, skeleton, vp):
            if (i != len(sr) - 2 and (vp[sr[i+1][3]] == 0 or __path_cost(skeleton, word, maxTransitionIndex, sr[i+1][1], sr[i+1][0]) > 10e20
                                      or not __does_baseline_exist(word, sr[i + 1], baseLine) and get_SHP(word, sr[i+1], baseLine, word.shape[0]) > get_SHP(word, sr[i+1], 0, baseLine)) and top_pixel_in_word_index + 16 >= __heighest_pixel_index(__horizintal_projection(word[0:word.shape[0], sr[i+1][3]:sr[i][3]])))\
                    or (i == len(sr)-2 and top_pixel_in_word_index + 12 >= __heighest_pixel_index(__horizintal_projection(word[0:word.shape[0], sr[i+1][3]:sr[i][3]])) and __does_hole_exist(word, [sr[i][3], sr[i+1][3]], maxTransitionIndex))\
                    or (i == len(sr) - 2 and top_pixel_in_word_index + 2 >= __heighest_pixel_index(__horizintal_projection(word[0:word.shape[0], sr[i + 1][3]:sr[i][3]]))):
                valid_sr.append(sr[i])
                i += 1

            elif (top_pixel_in_word_index != __heighest_pixel_index(__horizintal_projection(word[0:word.shape[0], sr[i+1][3]:sr[i][3]])) and (vp[sr[i+1][3]] == 0 or cost > 10e20)) \
                    or (top_pixel_in_word_index != __heighest_pixel_index(__horizintal_projection(word[0:word.shape[0], sr[i+1][0]:sr[i+1][1]]))and vp[sr[i+1][3]] == 0 and i + 1 == len(sr) - 1) or \
                    (not __does_baseline_exist(word, sr[i + 1], baseLine) and vp[sr[i + 1][3]] < most_frequent_value and vp[sr[i+1][3]] != 0 and __path_cost(skeleton, word, maxTransitionIndex, sr[i+1][1], sr[i+1][0]) < 10e20):
                
                i += 1
            else:
                valid_sr.append(sr[i])
                i += 1

        elif i < len(sr) - 1 and __is_stroke(word, sr[i + 1], sr[i], baseLine, top_pixel_in_word_index, most_frequent_value,
                                          most_frequent_value_after_0, maxTransitionIndex, second, skeleton, vp) and __does_dots_exist(word, sr[i + 1], sr[i]):
            valid_sr.append(sr[i])
            i += 1

        elif i < len(sr) - 1 and __is_stroke(word, sr[i + 1], sr[i], baseLine, top_pixel_in_word_index, most_frequent_value,
                                          most_frequent_value_after_0, maxTransitionIndex, second, skeleton, vp) and not __does_dots_exist(word, sr[i + 1], sr[i]):
            if i < len(sr) - 2 and __is_stroke(word, sr[i + 2], sr[i + 1], baseLine, top_pixel_in_word_index, most_frequent_value,
                                            most_frequent_value_after_0, maxTransitionIndex, second, skeleton, vp) and not __does_dots_exist(word, sr[i + 2], sr[i + 1]):
                valid_sr.append(sr[i])
                i += 3
            elif i < len(sr) - 3 and __is_stroke(word, sr[i + 2], sr[i + 1], baseLine, top_pixel_in_word_index, most_frequent_value,
                                              most_frequent_value_after_0, maxTransitionIndex, second, skeleton, vp) and __does_dots_exist(word, sr[i + 2], sr[i + 1])and __is_stroke(word, sr[i + 3], sr[i + 2], baseLine, top_pixel_in_word_index, most_frequent_value,
                                                                                                                                                                           most_frequent_value_after_0, maxTransitionIndex, second, skeleton, vp) and not __does_dots_exist(word, sr[i + 3], sr[i + 2]):
                valid_sr.append(sr[i])
                i += 3

            elif i < len(sr) - 2 and __heighest_pixel_index(__horizintal_projection(word[0:word.shape[0], sr[i+1][3]:sr[i][3]])) < maxTransitionIndex - 5 \
                and __heighest_pixel_index(
                    __horizintal_projection(word[0:word.shape[0], sr[i + 2][3]:sr[i + 1][3]])) > maxTransitionIndex - 2:
                valid_sr.append(sr[i])
                i += 1
            else:
                valid_sr.append(sr[i])
                i += 1

        else:
            valid_sr.append(sr[i])
            i += 1

    valid_sr.append(sr[len(sr) - 1])
    return valid_sr

def comp(x, y, mid, s, f, cond):
    ress = []

    if (cond == '=='):
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

def __cutting_points_identification(vp, word, mti, bl):
    i = 0
    flag = 0

    m = stats.mode(vp)
    mfv = m[0][0]
    p = [value for value in vp if value != 0]
    n = stats.mode(p)
    mfv1 = n[0][0]

    [StartIndex, EndIndex, MidIndex, CutIndex] = [0, 0, 0, 0]
    sr = []

    while (i < word.shape[1]):
        if word[mti, i] == 0 and flag == 0:
            EndIndex = i
            flag = 1
        elif word[mti, i] != 0 and flag == 1:
            StartIndex = i
            MidIndex = int((EndIndex + StartIndex) / 2)

            if (comp(vp, 0, MidIndex, StartIndex, EndIndex, '==') != False):
                CutIndex = comp(vp, 0, MidIndex, StartIndex, EndIndex, '==')
            elif vp[MidIndex] == mfv:
                CutIndex = MidIndex
            elif (comp(vp, mfv, MidIndex, MidIndex, EndIndex, '<=') != False):
                CutIndex = comp(vp, mfv, MidIndex, MidIndex, EndIndex, '<=')
            elif (comp(vp, mfv, MidIndex, StartIndex, MidIndex, '<=') != False):
                CutIndex = comp(vp, mfv, MidIndex, StartIndex, MidIndex, '<=')
            else:
                CutIndex = MidIndex

            sr.append([StartIndex, EndIndex, MidIndex, CutIndex])
            flag = 0
        
        i += 1
    return mfv1, mfv, sr

def __heighest_pixel_index(hp):
    i = 0
    while i < len(hp):
        if hp[i] != 0:
            break
        i += 1
    return i

def __cropping_indeces(hp, bl):
    start = 0
    i = bl
    while i >= 0:
        if hp[i] == 0 and i != bl:
            start = i
            break
        i -= 1

    end = len(hp) - 1
    i = bl
    while i <len(hp):
        if hp[i] == 0:
            end = i
            break
        i += 1
    return start, end

def __process_line(line):
    line = cv2.resize(line, (int(line.shape[1]*220/100), int(line.shape[0]*220/100)), interpolation=cv2.INTER_AREA)
    kernel = np.ones((2, 2), np.uint8)
    line1 = cv2.erode(line, kernel, iterations=2)
    ret, bw_img_line = cv2.threshold(line1, 120, 255, cv2.THRESH_BINARY)
    # get baseline using thinned img
    skeleton_line = skeletonize(bw_img_line - 255)
    bl_line = baseLine(__horizintal_projection(skeleton_line))
    # get maxTransition index
    maxTransitionIndex_line = __detect_maximum_transition(bw_img_line, bl_line)
    bw_img_line = line
    for i in range(bw_img_line.shape[0]):
        for j in range(bw_img_line.shape[1]):
            if bw_img_line[i,j] <= 190:
                bw_img_line[i,j] = 255
            else:
                bw_img_line[i,j] = 0
                
    most_frequent_value_after_0, most_frequent_value, cutting_points = __cutting_points_identification(__vertical_projection(bw_img_line), bw_img_line, maxTransitionIndex_line, bl_line)
    max_indeces_list = []
    cutting_points.reverse()
    for i in range(len(cutting_points) - 1):
        temp_img = line[0:line.shape[0], cutting_points[i + 1][3]:cutting_points[i][3]]
        start, end = __cropping_indeces(__horizintal_projection(temp_img), bl_line)
        if start not in max_indeces_list:
            max_indeces_list.append(start)
            
    max_indeces_list.sort()
    
    bw_img_line[bl_line, 0:bw_img_line.shape[1]] = 255
    
    return bl_line, maxTransitionIndex_line, bw_img_line, max_indeces_list[1]


def baseLine(hp):
 
    bl = 0
    maxBl = 0
    i = 1
    while i < len(hp):
        if maxBl < hp[i]:
            maxBl = hp[i]
            bl = i
        i += 1
    return bl

def __detect_maximum_transition(img, bl):
    maxTransition = 0
    maxTransitionIndex = bl
    for i in range(bl, 0, -1):
        currentTransition = 0
        flag = 0
        for j in range(img.shape[1]):
            if img[i,j] == 255 and flag == 0:
                currentTransition += 1
                flag = 1
            elif img[i,j] != 255 and flag == 1:
                flag = 0
        if currentTransition >= maxTransition:
            maxTransition = currentTransition
            maxTransitionIndex = i
    return maxTransitionIndex

def __cut_word(img, valid_cutting_points):
    chars = []
    for i in range(len(valid_cutting_points) - 1):
        if valid_cutting_points[i+1][3] == valid_cutting_points[i][3]:
            continue
        swap = img[0:img.shape[0], valid_cutting_points[i+1][3]:valid_cutting_points[i][3]]
        chars.append(swap)
    return chars

def __process_word(copy, img, line, bl_line, maxTransitionIndex_line, bw_img_line, second_peak):
    # convert img to binary where background is black
    img = cv2.resize(img, (int(img.shape[1]*220/100), int(img.shape[0]*220/100)), interpolation=cv2.INTER_AREA)
    copy = cv2.resize(copy, (int(copy.shape[1]*220/100), int(copy.shape[0]*220/100)), interpolation=cv2.INTER_AREA)
    
    kernel = np.ones((2, 2), np.uint8)
    img1 = cv2.erode(img, kernel, iterations=2)
    ret, bw_img = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY)
    # get baseline using thinned img
    skeleton = skeletonize(bw_img - 255)
    bl = baseLine(__horizintal_projection(skeleton))    
    # get maxTransition index
    maxTransitionIndex = __detect_maximum_transition(bw_img, bl)
    
    bw_img = img
    for i in range(bw_img.shape[0]):
        for j in range(bw_img.shape[1]):
            if bw_img[i,j] <= 165:
                bw_img[i,j] = 255
            else:
                bw_img[i,j] = 0
    most_frequent_value_after_0, most_frequent_value, cutting_points = __cutting_points_identification(__vertical_projection(bw_img), bw_img, maxTransitionIndex, bl_line) # bw_img_line

    # filter cutting points
    valid_sr = filter_cutting_points(skeleton, bw_img, cutting_points, bl_line, maxTransitionIndex, most_frequent_value, most_frequent_value_after_0, __vertical_projection(bw_img), __horizintal_projection(bw_img_line), second_peak)
   
    chars = __cut_word(copy, valid_sr)
    return chars

def segment_chars(lines, words):

    line_chars = []
    for i in range(len(lines)):

        bl_line, maxTransitionIndex_line, bw_img_line, second_peak = __process_line(lines[i])

        for word in words[i]:
            org = word
            letters = __process_word(org, word, lines[i], bl_line, maxTransitionIndex_line, bw_img_line, second_peak)
            line_chars.append(letters)

    return line_chars