import cv2
import numpy as np
from scipy import stats
from skimage.morphology import skeletonize

def detect_baseline(line):
    
    im = skeletonize(line- 255)
    hist = np.sum(im, 1)
    
    #6: HP <-smoothing(HP)
    
    #show_images([im])
    #print(hist)

    max_hist = 0
    for i in range(len(hist)):
        if hist[i] >= max_hist:
            max_hist = hist[i]
            baseline = i
    return baseline

def maximum_transition_line(line, baseline):
    max_transitions = 0
    max_transitions_index = baseline
    i = baseline
    
    width = line.shape[1] 
    height = line.shape[0]
    
    while (i < height):
        curr_transitions = 0
        flag = 0
        j = 0 
        while (j < width):
            if line[i, j] == 1 and flag == 0:
                curr_transitions += 1
                flag = 1
            elif line[i, j] != 0 and flag == 1:
                flag = 0
            j += 1
        if curr_transitions >= max_transitions:
            max_transitions = curr_transitions
            max_transitions_index = i
        i += 1
    return max_transitions_index

class SR:
    def __init__(self, ei = 0, si = 0, ci = 0, mi = 0):
        self.EndIndex = ei
        self.StartIndex = si
        self.CutIndex = ci
        self.MidIndex = mi

        
def comp(x, y, mid, s,f, cond):
    ress = []
    
    if (cond == '=='):
        for i in range(s,f):
            if x[i] == y:
                ress.append(i)
    else:
        for i in range(s,f):
            if x[i] <= y:
                ress.append(i)
    
    if len(ress)!=0:
        print(ress)
        val  = ress.flat[np.abs(ress - mid).argmin()]
        
        return val
    else:
        print(ress)
        return  False
    

def cpi(line, word, mti):
    i = 0
    flag = 0
    vp = np.sum(line, 0)
    #print(vp)
    
    m = stats.mode(vp)
    mfv = m[0][0]
    print(mfv)
    
    width = word.shape[1] 
    
    newSR = SR(0,0,0)
    sr = []
    while (i < width):
        if word[mti, i] == 1 and flag == 0:
            newSR.EndIndex = i 
            flag = 1
        elif word[mti, i] != 1 and flag == 1:
            newSR.StartIndex = i
            newSR.MidIndex = int((newSR.EndIndex + newSR.StartIndex)/2)
            
            if(comp(vp, 0 , newSR.MidIndex,newSR.StartIndex, newSR.EndIndex, '==') != False):
                newSR.CutIndex = comp(vp, 0 , newSR.MidIndex,newSR.StartIndex, newSR.EndIndex, '==')
            elif vp[newSR.MidIndex] == mfv:
                newSR.CutIndex = newSR.MidIndex
            elif (comp(vp, mfv , newSR.MidIndex,newSR.StartIndex, newSR.EndIndex, '<=') != False):
                newSR.CutIndex = comp(vp, mfv , newSR.MidIndex,newSR.StartIndex, newSR.EndIndex, '<=')
            elif (comp(vp, mfv , newSR.MidIndex,newSR.StartIndex, newSR.MidIndex, '<=') != False):
                newSR.CutIndex = comp(vp, mfv , newSR.MidIndex,newSR.StartIndex, newSR.MidIndex, '<=')
            else:
                newSR.CutIndex = newSR.MidIndex
                
            flag = 0    
        i += 1
    return newSR

img  = cv2.imread('line0.png', 0)
_, bw_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) 
baseline = detect_baseline(bw_img)
print(baseline)
mti = maximum_transition_line(bw_img, baseline)
print(mti)
wordim = cv2.imread('word0.png', 0)
sr = cpi(bw_img, wordim, mti)
print(sr.CutIndex, sr.MidIndex, sr.EndIndex, sr.StartIndex)