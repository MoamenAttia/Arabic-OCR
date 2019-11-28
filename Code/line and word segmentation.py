import cv2
import numpy as np

def lines(img):
    hist = np.sum(255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1)
    
    lines = []
    i = 0
    while (i < len(hist)):

        if (hist[i] != 0):
            x = i
            y = i
            while (y < len(hist)):
                y += 1
                if (y < len(hist)) and (hist[y] == 0):
                    break
            
            lines.append(img[x:y, 0:len(img[0])])
            i = y
        else:
            i += 1
            
    return lines

def words(lines):
    res = []
    kernel = np.ones((3,3),np.uint8)
    
    for j in range(len(lines)):

        img = np.rot90(lines[j])
        
        scale_percent = 220
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
        img2 = cv2.erode(img, kernel, iterations = 1) 
        img2 = cv2.dilate(img2, kernel, iterations = 1)
        
        hist = np.sum(255 - cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 1)
        i = 0
        while (i < len(hist)):
            th = 0
            
            if (hist[i] != 0):
                x = i
                y = i
                while (y < len(hist)) and (th <= 2):
                    y += 1
                    
                    if (y < len(hist)) and (hist[y] == 0):
                        th += 1
                
                res.append(img[x:y, 0:len(img[0])])
                i = y
            else:
                i += 1
                
    for k in range(len(res)):
        res[k] = (np.rot90(res[k], 3))
    
    return res 
        

img = cv2.imread('capr6.png')
lines = lines(img)
words = words(lines)
for j in range(len(words)):
  
    cv2.imwrite('word%d.png'%j, words[j])