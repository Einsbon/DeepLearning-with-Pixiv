import cv2
import sys
import os

ountside = 10

cascade_file = r"D:\machine_learning\lbpcascade_animeface-master\lbpcascade_animeface.xml"
cascade = cv2.CascadeClassifier(cascade_file)


def trim(filename, saveName):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     # scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(55, 55))
    if len(faces) == 0:
        return
    (x, y, w, h) = faces[0]

    x = int(x - w/8.0)
    y = int(y - h/8.0)
    w = int(w*1.25)
    h = w

    if x < 0:
        x = 0
    if y < 0:
        y = 0

    img_trim = image[y:y+h, x:x+w]
    #cv2.imshow("asdf", img_trim)
    # cv2.waitKey(0)
    cv2.imwrite(saveName, img_trim)


imagesPath = r'D:\picture\miku1000'
savePath = r'D:\picture\miku_square3'
imagesList = os.listdir(imagesPath)
imagesList.sort()

for imgPath in imagesList:
    trim(imagesPath+'\\'+imgPath, savePath+'\\'+imgPath)
