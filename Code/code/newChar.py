import cv2
import numpy as np
from skimage.morphology import skeletonize
from cutting_points import cutting_points_identification
from filter_cutting_points import filter_cutting_points

def horizintal_projection(im):
    #im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    projection = np.sum(im, 1)  # Calculate horizontal projection
    return projection

def vertical_projection(im):
    projection = np.sum(im, 0)
    return projection



def get_heighest_pixel_index(hp):
    i = 0
    while i < len(hp):
        if hp[i] != 0:
            break
        i += 1
    return i




def get_stroke_cropping_indeces(hp, bl):
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



def get_line_info(line):
    #                               FOR LINE
    # convert img to binary where background is black
    line = cv2.resize(line, (int(line.shape[1]*220/100), int(line.shape[0]*220/100)), interpolation=cv2.INTER_AREA)
    kernel = np.ones((2, 2), np.uint8)
    line1 = cv2.erode(line, kernel, iterations=2)
    ret, bw_img_line = cv2.threshold(line1, 120, 255, cv2.THRESH_BINARY)
    # get baseline using thinned img
    skeleton_line = skeletonize(bw_img_line - 255)
    bl_line = baseLine(horizintal_projection(skeleton_line))
    # get maxTransition index
    maxTransitionIndex_line = max_transitions(bw_img_line, bl_line)
    bw_img_line = line
    for i in range(bw_img_line.shape[0]):
        for j in range(bw_img_line.shape[1]):
            if bw_img_line[i,j] <= 190:
                bw_img_line[i,j] = 255
            else:
                bw_img_line[i,j] = 0
                
    most_frequent_value_after_0, most_frequent_value, cutting_points = cutting_points_identification(vertical_projection(bw_img_line), bw_img_line, maxTransitionIndex_line, bl_line)
    max_indeces_list = []
    cutting_points.reverse()
    for i in range(len(cutting_points) - 1):
        temp_img = line[0:line.shape[0], cutting_points[i + 1][3]:cutting_points[i][3]]
        start,end = get_stroke_cropping_indeces(horizintal_projection(temp_img), bl_line)
        #temp_hp = horizintal_projection(line[0:line.shape[0], cutting_points[i][1]:cutting_points[i][0]])
        if start not in max_indeces_list:
            max_indeces_list.append(start)
            
    max_indeces_list.sort()
    #print(max_indeces_list)
    
    
    bw_img_line[bl_line, 0:bw_img_line.shape[1]] = 255
    
    return bl_line, maxTransitionIndex_line, bw_img_line, max_indeces_list[1]


def baseLine(hp):
    #bl = np.where(hp == np.amax(hp))
    #return int(bl[0])
    bl = 0
    maxBl = 0
    i = 1
    while i < len(hp):
        if maxBl < hp[i]:
            maxBl = hp[i]
            bl = i
        i += 1
    return bl

def max_transitions(img, bl):
    maxTransition = 0
    maxTransitionIndex = bl
    #print(img.shape[0],img.shape[1])
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
    # img[maxTransitionIndex,0:20] = 1
    #img[0:5,0:10] = 1
    # print(maxTransitionIndex)
    # show_images([img])
    return maxTransitionIndex


def cut_chars(blk, bw_img, valid_cutting_points, bl_line):
    chars = []
    for i in range(len(valid_cutting_points) - 1):
        if valid_cutting_points[i+1][3] == valid_cutting_points[i][3]:
            continue
        swap = blk[0:blk.shape[0], valid_cutting_points[i+1][3]:valid_cutting_points[i][3]]
        #swap = cv2.resize(swap, (50, 50), interpolation=cv2.INTER_AREA)

        chars.append(swap)
        
    #for i in range(len(chars)):
    #        chars[i] = cv2.resize(chars[i], (int(chars[i].shape[1]*100/220), int(chars[i].shape[0]*100/220)), interpolation=cv2.INTER_AREA)
    #        chars[i] = cv2.resize(chars[i], (int(chars[i].shape[1]*400/100), int(chars[i].shape[0]*400/100)), interpolation=cv2.INTER_AREA)
    #        chars[i] = cv2.resize(chars[i], (50,50), interpolation=cv2.INTER_AREA)

    return chars

# main
def get_chars2(blk, img, line, bl_line, maxTransitionIndex_line, bw_img_line, second_peak):
    #                               FOR WORD
    # convert img to binary where background is black
    img = cv2.resize(img, (int(img.shape[1]*220/100), int(img.shape[0]*220/100)), interpolation=cv2.INTER_AREA)
    blk = cv2.resize(blk, (int(blk.shape[1]*220/100), int(blk.shape[0]*220/100)), interpolation=cv2.INTER_AREA)
    
    kernel = np.ones((2, 2), np.uint8)
    img1 = cv2.erode(img, kernel, iterations=2)
    ret, bw_img = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY)
    # get baseline using thinned img
    skeleton = skeletonize(bw_img - 255)
    bl = baseLine(horizintal_projection(skeleton))    
    # get maxTransition index
    maxTransitionIndex = max_transitions(bw_img, bl)
    
    # get cutting points candidates
    bw_img = img
    for i in range(bw_img.shape[0]):
        for j in range(bw_img.shape[1]):
            if bw_img[i,j] <= 165:
                bw_img[i,j] = 255
            else:
                bw_img[i,j] = 0
    most_frequent_value_after_0, most_frequent_value, cutting_points = cutting_points_identification(vertical_projection(bw_img), bw_img, maxTransitionIndex, bl_line)
    # print( most_frequent_value_after_0, most_frequent_value)

    # filter cutting points
    valid_cutting_points = filter_cutting_points(skeleton, bw_img, cutting_points, bl_line, maxTransitionIndex, most_frequent_value, most_frequent_value_after_0, vertical_projection(bw_img), horizintal_projection(bw_img_line), second_peak)
    
    #bw_img_line[bl_line, 0:bw_img_line.shape[1]] = 255
    #cv2.imwrite('ziko.png', blk)

    # cut letters
    chars = cut_chars(blk, bw_img, valid_cutting_points, bl_line)
    return chars
    #print(len(chars))
    #for i in range(len(chars)):
     #   str1 = 'c'+str(i)+'.png'
      #  cv2.imwrite(str1,chars[i])
      
def get_characters(lines, words):
    #original = cv2.imread('w47.png', 0)
    #line = cv2.imread('l3.png', 0)
    #bl_line, maxTransitionIndex_line, bw_img_line, second_peak = get_line_info(line)
    #get_chars2(original, line, bl_line, maxTransitionIndex_line, bw_img_line, second_peak)
    
    chars_list = []
    
    #bl_line, maxTransitionIndex_line, bw_img_line, second_peak = get_line_info(lines[0])
    #print(bl_line, maxTransitionIndex_line, bw_img_line, second_peak )
    #word = get_chars2(words[0][0], lines[0], bl_line, maxTransitionIndex_line, bw_img_line, second_peak)
    #print(words[0][0].shape)
    #print(lines[0].shape)
    #x = lines[0]
    #x[bl_line, 0:x.shape[1]] = 255
    #cv2.imwrite('ziko.png', words[0][0])
    
    for i in range(len(lines)):
        #if i > 2 and i < len(lines) - 2:
        #    continue
        bl_line, maxTransitionIndex_line, bw_img_line, second_peak = get_line_info(lines[i])
        line_chars = []
        for j in range(len(words[i])):
            blk = words[i][j]
            word = get_chars2(blk, words[i][j], lines[i], bl_line, maxTransitionIndex_line, bw_img_line, second_peak)
            line_chars.append(word)

        chars_list.append(line_chars)
    
    return chars_list