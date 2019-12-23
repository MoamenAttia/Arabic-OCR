import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy import stats
import skimage.graph

def __horizintal_projection(im): 
    return np.sum(im, axis=1) 

def __does_hole_exist(word, boundaries, mti):
    img = word[0:mti, 0:word.shape[1]]
    vp = np.sum(img, axis=0)

    for i in range(boundaries[1], boundaries[0]):
        if vp[i] == 0:
            return False
    return True

def __does_baseline_exist(word, boundaries, bl):
    for i in range(boundaries[1], boundaries[3]):
        if word[bl, i] == 0:
            return False
    return True

def __cal_SHP(word, boundaries, start, end):
    img = word[start:end, boundaries[1]:boundaries[0]]
    hp_img = __horizintal_projection(img)
    return np.sum(hp_img)

def __highest_left_pixel(word, s, f):
    img = word[0:word.shape[0], f[3]:s[3]]
    vp = np.sum(img, axis=0)
    i = 0
    while vp[i] == 0:
        i += 1
    j = 0
    while j < word.shape[0]:
        if img[j, i] != 0:
            return j
        j += 1
    return word.shape[0]

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

def __is_connected_component(img, bl):
    [cnt, flag] = [0, 0]
    for i in range(img.shape[1]):
        if img[bl, i] == 255 and flag == 0:
            cnt += 1
            flag = 1
        elif img[bl, i] != 255 and flag == 1:
            flag = 0
    return cnt > 1

def __is_stroke(word, s, f, bl, heighest_pixel, mfv, second_mfv, mti, second, skeleton, vp):
    img = word[0:word.shape[0], s[3]:f[3]]
    start, end = __cropping_indeces(__horizintal_projection(img), bl)
    img1 = word[start:end, s[3]:f[3]]

    if __is_connected_component(img, int(word.shape[0]/2)):
        return False

    if __cal_SHP(img1, [img1.shape[1], 0], 0, bl) <= __cal_SHP(img1, [img1.shape[1], 0], bl, img1.shape[0]):
        return False

    hp_img = __horizintal_projection(img1)
    height_of_seg = bl - start
    second_peak = bl - second
    if height_of_seg > second_peak or start <= heighest_pixel + 2:
        return False

    hp_img = __horizintal_projection(img1)
    hp_img = [value for value in hp_img if value != 0]
    m = stats.mode(hp_img)
    mode_hp = m[0][0]
    if mfv == 0:
        mfv = second_mfv
    if mode_hp not in range(mfv - 600, mfv + 600):
        return False

    if (__does_hole_exist(img, [img.shape[1], 0], mti) and not __does_dots_exist(word, s, f)) or __does_hole_exist(word, s, mti) or __path_cost(skeleton, word, mti, s[1], s[0]) > 10e20 or vp[s[3]] == 0:
        return False
    return True

def __does_dots_exist(word, s, e):
    img = word[0:word.shape[0], s[3] + 2:e[3] - 2]
    hp = __horizintal_projection(img)
    [count, flag] = [0, 0]
    for j in range(len(hp)):
        if hp[j] > 0 and flag == 0:
            count += 1
            flag = 1
        elif hp[j] == 0 and flag == 1:
            flag = 0

    if count > 1:
        return True
    return False

def __filter_separation_region(skeleton, word, sr_list, baseline, maxTransitionIndex, mfv, second_mfv, vp, hp, second):
    i = 0
    sr_list.append([word.shape[1] - 1, word.shape[1] - 1,
               word.shape[1] - 1, word.shape[1] - 1])
    valid_sr = []
    heighest_pixel = __heighest_pixel_index(hp)
    sr_list.reverse()

    if i < len(sr_list) - 1 and __is_stroke(word, sr_list[i + 1], sr_list[i], baseline, heighest_pixel, mfv,
                                    second_mfv, maxTransitionIndex, second, skeleton,
                                    vp) and not __does_dots_exist(word, sr_list[i + 1], sr_list[i]):
        if i < len(sr_list) - 2 and __is_stroke(word, sr_list[i + 2], sr_list[i + 1], baseline, heighest_pixel,
                                        mfv,
                                        second_mfv, maxTransitionIndex, second, skeleton,
                                        vp) and not __does_dots_exist(word, sr_list[i + 2], sr_list[i + 1]):
            valid_sr.append(sr_list[i])
            i += 3
        elif i < len(sr_list) - 3 and __is_stroke(word, sr_list[i + 2], sr_list[i + 1], baseline, heighest_pixel,
                                          mfv,
                                          second_mfv, maxTransitionIndex, second, skeleton,
                                          vp) and __does_dots_exist(word, sr_list[i + 2], sr_list[i + 1]) and __is_stroke(word, sr_list[i + 3],
                                                                                                     sr_list[i + 2],
                                                                                                     baseline,
                                                                                                     heighest_pixel,
                                                                                                     mfv,
                                                                                                     second_mfv,
                                                                                                     maxTransitionIndex,
                                                                                                     second, skeleton,
                                                                                                     vp) and not __does_dots_exist(
                word, sr_list[i + 3], sr_list[i + 2]):
            valid_sr.append(sr_list[i])
            i += 3

        else:
            i += 1


    valid_sr.append(sr_list[0])
    while i < len(sr_list) - 1:
        cost = __path_cost(
            skeleton, word, maxTransitionIndex, sr_list[i][1], sr_list[i][0])

        if vp[sr_list[i][3]] == 0:
            if i < len(sr_list) - 1 and __is_stroke(word, sr_list[i + 1], sr_list[i], baseline, heighest_pixel,
                                            mfv,
                                            second_mfv, maxTransitionIndex, second, skeleton,
                                            vp) and not __does_dots_exist(word, sr_list[i + 1], sr_list[i]):
                if i < len(sr_list) - 2 and __is_stroke(word, sr_list[i + 2], sr_list[i + 1], baseline, heighest_pixel,
                                                mfv,
                                                second_mfv, maxTransitionIndex, second, skeleton,
                                                vp) and not __does_dots_exist(word, sr_list[i + 2], sr_list[i + 1]):
                    valid_sr.append(sr_list[i])
                    i += 3
                elif i < len(sr_list) - 3 and __is_stroke(word, sr_list[i + 2], sr_list[i + 1], baseline, heighest_pixel,
                                                  mfv,
                                                  second_mfv, maxTransitionIndex, second, skeleton,
                                                  vp) and __does_dots_exist(word, sr_list[i + 2], sr_list[i + 1]) and __is_stroke(word,
                                                                                                             sr_list[i + 3],
                                                                                                             sr_list[i + 2],
                                                                                                             baseline,
                                                                                                             heighest_pixel,
                                                                                                             mfv,
                                                                                                             second_mfv,
                                                                                                             maxTransitionIndex,
                                                                                                             second,
                                                                                                             skeleton,
                                                                                                             vp) and not __does_dots_exist(
                        word, sr_list[i + 3], sr_list[i + 2]):
                    valid_sr.append(sr_list[i])
                    i += 3
                elif __heighest_pixel_index(__horizintal_projection(word[0:word.shape[0], sr_list[i+1][3]:sr_list[i][3]])) < maxTransitionIndex - 5:
                    valid_sr.append(sr_list[i])
                    i += 1
                else:
                    i += 1
            else:
                valid_sr.append(sr_list[i])
                i += 1

        elif cost > 10e20:
            if i < len(sr_list) - 1 and __is_stroke(word, sr_list[i + 1], sr_list[i], baseline, heighest_pixel,
                                            mfv,
                                            second_mfv, maxTransitionIndex, second, skeleton,
                                            vp) and not __does_dots_exist(word, sr_list[i + 1], sr_list[i]):
                if i < len(sr_list) - 2 and __is_stroke(word, sr_list[i + 2], sr_list[i + 1], baseline, heighest_pixel,
                                                mfv,
                                                second_mfv, maxTransitionIndex, second, skeleton,
                                                vp) and not __does_dots_exist(word, sr_list[i + 2], sr_list[i + 1]):
                    valid_sr.append(sr_list[i])
                    i += 3
                elif i < len(sr_list) - 3 and __is_stroke(word, sr_list[i + 2], sr_list[i + 1], baseline, heighest_pixel,
                                                  mfv,
                                                  second_mfv, maxTransitionIndex, second, skeleton,
                                                  vp) and __does_dots_exist(word, sr_list[i + 2], sr_list[i + 1]) and __is_stroke(word,
                                                                                                             sr_list[i + 3],
                                                                                                             sr_list[i + 2],
                                                                                                             baseline,
                                                                                                             heighest_pixel,
                                                                                                             mfv,
                                                                                                             second_mfv,
                                                                                                             maxTransitionIndex,
                                                                                                             second,
                                                                                                             skeleton,
                                                                                                             vp) and not __does_dots_exist(
                        word, sr_list[i + 3], sr_list[i + 2]):
                    valid_sr.append(sr_list[i])
                    i += 3
                elif __heighest_pixel_index(__horizintal_projection(word[0:word.shape[0], sr_list[i+1][3]:sr_list[i][3]])) < maxTransitionIndex - 5:
                    valid_sr.append(sr_list[i])
                    i += 1

                else:

                    i += 1
            else:
                valid_sr.append(sr_list[i])
                i += 1

        elif __does_hole_exist(word, sr_list[i], maxTransitionIndex):
            i += 1

        elif not __does_baseline_exist(word, sr_list[i], baseline):
            if __cal_SHP(word, sr_list[i], baseline, word.shape[0]) > __cal_SHP(word, sr_list[i], 0, baseline):
                i += 1

            elif vp[sr_list[i][3]]/255 < mfv/255:
                valid_sr.append(sr_list[i])
                i += 1
            else:
                i += 1

        elif (vp[sr_list[i + 1][3]] == 0 or i == len(sr_list) - 2) and (-__highest_left_pixel(word, sr_list[i], sr_list[i + 1]) + baseline) < int((-heighest_pixel + baseline)/2)\
                and __path_cost(skeleton, word, maxTransitionIndex, sr_list[i][1], sr_list[i][0]) < 10e20 and vp[sr_list[i+1][3]] != 0:
            i += 1

        elif i < len(sr_list) - 1 and not __is_stroke(word, sr_list[i + 1], sr_list[i], baseline, heighest_pixel, mfv, second_mfv, maxTransitionIndex, second, skeleton, vp):
            if (i != len(sr_list) - 2 and (vp[sr_list[i+1][3]] == 0 or __path_cost(skeleton, word, maxTransitionIndex, sr_list[i+1][1], sr_list[i+1][0]) > 10e20
                                      or not __does_baseline_exist(word, sr_list[i + 1], baseline) and __cal_SHP(word, sr_list[i+1], baseline, word.shape[0]) > __cal_SHP(word, sr_list[i+1], 0, baseline)) and heighest_pixel + 16 >= __heighest_pixel_index(__horizintal_projection(word[0:word.shape[0], sr_list[i+1][3]:sr_list[i][3]])))\
                    or (i == len(sr_list)-2 and heighest_pixel + 12 >= __heighest_pixel_index(__horizintal_projection(word[0:word.shape[0], sr_list[i+1][3]:sr_list[i][3]])) and __does_hole_exist(word, [sr_list[i][3], sr_list[i+1][3]], maxTransitionIndex))\
                    or (i == len(sr_list) - 2 and heighest_pixel + 2 >= __heighest_pixel_index(__horizintal_projection(word[0:word.shape[0], sr_list[i + 1][3]:sr_list[i][3]]))):
                valid_sr.append(sr_list[i])
                i += 1

            elif (heighest_pixel != __heighest_pixel_index(__horizintal_projection(word[0:word.shape[0], sr_list[i+1][3]:sr_list[i][3]])) and (vp[sr_list[i+1][3]] == 0 or cost > 10e20)) \
                    or (heighest_pixel != __heighest_pixel_index(__horizintal_projection(word[0:word.shape[0], sr_list[i+1][0]:sr_list[i+1][1]]))and vp[sr_list[i+1][3]] == 0 and i + 1 == len(sr_list) - 1) or \
                    (not __does_baseline_exist(word, sr_list[i + 1], baseline) and vp[sr_list[i + 1][3]] < mfv and vp[sr_list[i+1][3]] != 0 and __path_cost(skeleton, word, maxTransitionIndex, sr_list[i+1][1], sr_list[i+1][0]) < 10e20):
                
                i += 1
            else:
                valid_sr.append(sr_list[i])
                i += 1

        elif i < len(sr_list) - 1 and __is_stroke(word, sr_list[i + 1], sr_list[i], baseline, heighest_pixel, mfv,
                                          second_mfv, maxTransitionIndex, second, skeleton, vp) and __does_dots_exist(word, sr_list[i + 1], sr_list[i]):
            valid_sr.append(sr_list[i])
            i += 1

        elif i < len(sr_list) - 1 and __is_stroke(word, sr_list[i + 1], sr_list[i], baseline, heighest_pixel, mfv,
                                          second_mfv, maxTransitionIndex, second, skeleton, vp) and not __does_dots_exist(word, sr_list[i + 1], sr_list[i]):
            if i < len(sr_list) - 2 and __is_stroke(word, sr_list[i + 2], sr_list[i + 1], baseline, heighest_pixel, mfv,
                                            second_mfv, maxTransitionIndex, second, skeleton, vp) and not __does_dots_exist(word, sr_list[i + 2], sr_list[i + 1]):
                valid_sr.append(sr_list[i])
                i += 3
            elif i < len(sr_list) - 3 and __is_stroke(word, sr_list[i + 2], sr_list[i + 1], baseline, heighest_pixel, mfv,
                                              second_mfv, maxTransitionIndex, second, skeleton, vp) and __does_dots_exist(word, sr_list[i + 2], sr_list[i + 1])and __is_stroke(word, sr_list[i + 3], sr_list[i + 2], baseline, heighest_pixel, mfv,
                                                                                                                                                                           second_mfv, maxTransitionIndex, second, skeleton, vp) and not __does_dots_exist(word, sr_list[i + 3], sr_list[i + 2]):
                valid_sr.append(sr_list[i])
                i += 3

            elif i < len(sr_list) - 2 and __heighest_pixel_index(__horizintal_projection(word[0:word.shape[0], sr_list[i+1][3]:sr_list[i][3]])) < maxTransitionIndex - 5 \
                and __heighest_pixel_index(
                    __horizintal_projection(word[0:word.shape[0], sr_list[i + 2][3]:sr_list[i + 1][3]])) > maxTransitionIndex - 2:
                valid_sr.append(sr_list[i])
                i += 1
            else:
                valid_sr.append(sr_list[i])
                i += 1

        else:
            valid_sr.append(sr_list[i])
            i += 1

    valid_sr.append(sr_list[len(sr_list) - 1])
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
    _mfv = n[0][0]

    [StartIndex, EndIndex, MidIndex, CutIndex] = [0, 0, 0, 0]
    sr_list = []

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

            sr_list.append([StartIndex, EndIndex, MidIndex, CutIndex])
            flag = 0
        
        i += 1
    return _mfv, mfv, sr_list

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

def __detect_baseline(hp):
    return np.argmax(hp)

def __detect_maximum_transition(img, bl):
    maxTransition = 0
    mti = bl
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
            mti = i
    return mti

def __process_line(line):
    line = cv2.resize(line, (int(line.shape[1]*220/100), int(line.shape[0]*220/100)), interpolation=cv2.INTER_AREA)
    kernel = np.ones((2, 2), np.uint8)
    line1 = cv2.erode(line, kernel, iterations=2)
    ret, bw_line = cv2.threshold(line1, 120, 255, cv2.THRESH_BINARY)
    skeleton_line = skeletonize(bw_line - 255)
    bl_line = __detect_baseline(__horizintal_projection(skeleton_line))
    maxTransitionIndex_line = __detect_maximum_transition(bw_line, bl_line)
    bw_line = line
    for i in range(bw_line.shape[0]):
        for j in range(bw_line.shape[1]):
            if bw_line[i,j] <= 190:
                bw_line[i,j] = 255
            else:
                bw_line[i,j] = 0
                
    vp = np.sum(bw_line, axis=0)            
    _, _, cutting_points = __cutting_points_identification(vp, bw_line, maxTransitionIndex_line, bl_line)
    max_indeces_list = []
    cutting_points.reverse()
    for i in range(len(cutting_points) - 1):
        temp_img = line[0:line.shape[0], cutting_points[i + 1][3]:cutting_points[i][3]]
        start, end = __cropping_indeces(__horizintal_projection(temp_img), bl_line)
        if start not in max_indeces_list:
            max_indeces_list.append(start)
            
    max_indeces_list.sort()
    
    bw_line[bl_line, 0:bw_line.shape[1]] = 255
    
    return bl_line, maxTransitionIndex_line, bw_line, max_indeces_list[1]

def __cut_word(img, valid_cutting_points):
    chars = []
    for i in range(len(valid_cutting_points) - 1):
        if valid_cutting_points[i+1][3] == valid_cutting_points[i][3]:
            continue
        lt = img[0:img.shape[0], valid_cutting_points[i+1][3]:valid_cutting_points[i][3]]
        chars.append(lt)
    return chars

def __process_word(img, line, bl_line, maxTransitionIndex_line, bw_line, second_peak):
    img = cv2.resize(img, (int(img.shape[1]*220/100), int(img.shape[0]*220/100)), interpolation=cv2.INTER_AREA)
    _im = np.asarray(img, dtype=float)
    copy = _im.copy()
    
    kernel = np.ones((2, 2), np.uint8)
    img1 = cv2.erode(img, kernel, iterations=2)
    ret, bw_img = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY)
    skeleton = skeletonize(bw_img - 255)
    bl = __detect_baseline(__horizintal_projection(skeleton))    
    maxTransitionIndex = __detect_maximum_transition(bw_img, bl)
    
    bw_img = img
    for i in range(bw_img.shape[0]):
        for j in range(bw_img.shape[1]):
            if bw_img[i,j] <= 165:
                bw_img[i,j] = 255
            else:
                bw_img[i,j] = 0

    vp_word = np.sum(bw_img, axis=0)
    vp_line = np.sum(bw_line, axis=0)
    second_mfv, mfv, cutting_points = __cutting_points_identification(vp_word, bw_img, maxTransitionIndex, bl_line)

    valid_sr = __filter_separation_region(skeleton, bw_img, cutting_points, bl_line, maxTransitionIndex, mfv, second_mfv, vp_word, __horizintal_projection(bw_line), second_peak)
   
    chars = __cut_word(copy, valid_sr)
    return chars

def segment_chars(lines, words):

    chars = []
    for i in range(len(lines)):

        baseline, mti, bw_line, second_peak = __process_line(lines[i])

        for word in words[i]:
            org = word
            letters = __process_word(word, lines[i], baseline, mti, bw_line, second_peak)
            chars.append(letters)

    return chars