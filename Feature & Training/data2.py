import os
import cv2


folder = 'zeen'
path = 'C:\\Users\\medo\\Desktop\\pattern\\dataset\\' + folder

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.png' in file:
            files.append(os.path.join(r, file))


width = 16
height = 16
dim = (width, height)
for i, f in enumerate(files):
    img = cv2.imread(f)
    #img = RemovePadding(img)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    if i == 0:
        os.mkdir(folder)
    cv2.imwrite(folder+'\%d.png' % i, resized)

    print(f)
