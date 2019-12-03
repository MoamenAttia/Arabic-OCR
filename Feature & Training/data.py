import os
import cv2
import numpy as np


def RemovePadding(img):  # takes image and removes empty space from left and right
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    colarr_temp = np.sum(binary, axis=1)
    i, up = 0, False
    j, down = len(colarr_temp)-1, False
    while True:
        if colarr_temp[i] == 0:
            i = i+1
            binary = np.delete(binary, 0, axis=0)
        else:
            up = True
        if colarr_temp[j] == 0:
            j = j-1
            binary = np.delete(binary, -1, axis=0)
        else:
            down = True
        if up and down:
            break
    return binary


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def main():

    dirName = 'C:/Users\Mido\Desktop\pattern\dataset\noon'

    # Get the list of all files in directory tree at given path

    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    width = 16
    height = 16
    dim = (width, height)

# resize image

    # Print the files
    for i, elem in enumerate(listOfFiles):
        img = cv2.imread(elem)
        #img = RemovePadding(img)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite('noon\%d.png' % i, resized)


if __name__ == '__main__':
    main()
