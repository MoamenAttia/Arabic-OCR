import numpy as np
import cv2
import math
#####################################################
MAXCHARLEFT = 7
MAXCHARRIGHT = 6
MAXCHARMIDDLE = 8
############################


def skew(image):#takes image and fixes skewed text 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return rotated
####################################################
def RemovePadding(img): #takes image and removes empty space from left and right
    colarr_temp = np.sum(img, axis=0)
    i, left = 0, False
    j, right = len(colarr_temp)-1, False
    while True:
        if colarr_temp[i] == 0:
            i = i+1
            img = np.delete(img, 0, axis=1)
        else:
            left = True
        if colarr_temp[j] == 0:
            j = j-1
            img = np.delete(img, -1, axis=1)
        else:
            right = True
        if left and right:
            break
    return img
####################################################
def PreProcess(img): #takes the name of the image to load and convert to binary and returns the image
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    ret, img = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
    #img = cv2.bitwise_not(blur)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = RemovePadding(img)
    return img
################################################
def CutOnZeroes(img):#cuts image into multiple images #the cuts happen on the empty spaces between characters
    cv2.imwrite('final%d2.png', img)
    images = []
    cuts = []
    colarr = np.sum(img, axis=0)
    indices = [i for i, x in enumerate(colarr) if x == 0]
    i = 0
    for i in range(len(indices)-1):
        if indices[i+1]-indices[i] < 4 and indices[i] != 0 and indices[i+1] != 0:
            indices[i] = 0

    indices = list(filter(lambda a: a != 0, indices))
    indices.append(img.shape[1]-1)
    print(indices)
    j = 0
    for i in range(len(indices)):
        newimage = RemovePadding(img[:, j:indices[i]])
        cv2.imwrite("p%d.png" % i, newimage)
        images.append(newimage)
        j = indices[i]
        print(j)
    return images


########################################################
img = PreProcess("word8.png")
#parts = CutOnZeroes(img)
# print(imgarr)
cv2.waitKey(0)
# calculate sum over column and minumum
# segment on the zeros
# calculate hist on column

cuts = [[0, 0]]
img = img / 255
colarr = np.sum(img, axis=0)
colavg = np.sum(colarr) / len(colarr)

colarr_unique = list(set(colarr))
colarr_unique.sort()
if 0 in colarr_unique:
    minumum = colarr_unique[1]
else:
    minumum = colarr_unique[0]


# claculate baseline index
rowarr = list(np.sum(img, axis=1))
baseline = rowarr.index(max(rowarr))

first_half = img[0:baseline, :]
second_half = img[baseline+1:img.shape[0]-1, :]


cv2.imshow('g', first_half)
cv2.imwrite('final%d2.png', second_half)

cv2.waitKey(0)
'''
# get two consecutive minumum
for i in range(len(colarr)):
    if colarr[i] == 0:
        #cuts.append([i, 2])
        continue

    if colarr[i] <= math.ceil(0.4*colavg) and colarr[i+1] <= math.ceil(0.4*colavg) and i > 0:
        cuts.append([i, 1])


cuts.append((len(colarr)-1, 0))

cv2.imshow("ay", img)

for i in range(len(cuts)):
    img[:, cuts[i][0]] = 0


cv2.imshow("wla", img)

cv2.imwrite('final%d2.png', img)
print(cuts)
for i in range(len(cuts)-1):
    if (cuts[i+1][0] - cuts[i][0] < MAXCHARLEFT and i == 0 and cuts[i][1] != 2):
        cuts[i+1][1] = 0
    elif (cuts[i+1][0] - cuts[i][0] < MAXCHARMIDDLE and i < len(cuts)-1 and cuts[i][1] != 2):
        cuts[i+1][1] = 0
    elif (cuts[i+1][0] - cuts[i][0] < MAXCHARRIGHT and i == len(cuts)-1 and cuts[i][1] != 2):
        cuts[i][1] = 0
    elif (cuts[i+1][0] - cuts[i][0] < 2 and cuts[i][1] == 2):
        cuts[i][1] = 0

for i in range(len(cuts)):
    if cuts[i][1] != 0:
        print(cuts[i][0])
'''
