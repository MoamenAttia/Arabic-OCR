import numpy as np
from scipy import stats
import cv2

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


def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

def cutting_points_identification(vp, word, mti, bl):
    i = 0
    flag = 0

    m = stats.mode(vp)
    mfv = m[0][0]
    p = remove_values_from_list(vp, 0)
    n = stats.mode(p)
    mfv1 = n[0][0]
    #print(mfv1)
    #comp_img = word[0:mti, 0:word.shape[1]]
    
    StartIndex = 0
    EndIndex = 0
    MidIndex = 0
    CutIndex = 0
    sr = []

    while (i < word.shape[1]):
        if word[mti, i] == 0 and flag == 0:
            EndIndex = i
            flag = 1
        elif word[mti, i] != 0 and flag == 1:
            StartIndex = i
            MidIndex = int((EndIndex + StartIndex) / 2)

            if (comp(vp, 0, MidIndex, StartIndex, EndIndex, '==') != False):
                print('cond1')
                CutIndex = comp(vp, 0, MidIndex, StartIndex, EndIndex, '==')
            elif vp[MidIndex] == mfv:
                print('cond2')
                CutIndex = MidIndex
            elif (comp(vp, mfv, MidIndex, MidIndex, EndIndex, '<=') != False):
                print('cond3')
                CutIndex = comp(vp, mfv, MidIndex, MidIndex, EndIndex, '<=')
            elif (comp(vp, mfv, MidIndex, StartIndex, MidIndex, '<=') != False):
                print('cond4')
                CutIndex = comp(vp, mfv, MidIndex, StartIndex, MidIndex, '<=')
            else:
                print('cond5')
                CutIndex = MidIndex

            #print(StartIndex,' ', EndIndex,' ', MidIndex,' ', CutIndex)
            sr.append([StartIndex, EndIndex, MidIndex, CutIndex])
            flag = 0
        
        i += 1

    #word[mti, 0:word.shape[1]] = 255 
    #word[bl, 0:word.shape[1]] = 255 
    #for i in range(len(sr)):
    #    word[0:word.shape[0],sr[i][1]] = 255 
    #    word[0:word.shape[0],sr[i][0]] = 255 

       
    #cv2.imwrite('zapateria.png', word)
    return mfv1, mfv, sr