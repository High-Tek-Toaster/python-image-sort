import os
import ntpath
import shutil
import cv2


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def MoveBlurryImages(temp, storagePath, value):
    pictureList = []
    for dirpath, dirnames, filenames in os.walk(storagePath):
        for file in filenames:
            if file.endswith('.jpg'):
                pictureList.append(file)

    os.mkdir(temp)

    for img in pictureList:
        pic = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        if cv2.Laplacian(pic, cv2.CV_64F.var()) < value:
            shutil.move(img, temp + path_leaf(img))

def LaplacianValue(img):
    return cv2.Laplacian(cv2.imread(img, cv2.IMREAD_GRAYSCALE), cv2.CV_64F.var())