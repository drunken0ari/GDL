import cv2 as cv
from cv2 import aruco
import numpy as np
import math
import os
from os import listdir
from os.path import isfile, join

def Project(points, intrinsic, distortion):
    result = []
    rvec = tvec = np.array([0.0, 0.0, 0.0])
    if len(points) > 0:
        result, _ = cv.projectPoints(points, rvec, tvec, intrinsic, distortion)
    return np.squeeze(result, axis=1)

dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
parameters = cv.aruco.DetectorParameters_create()

#cap = cv.VideoCapture(0) #give the server id shown in IP webcam App

mtxe = [[  391.05,   0.00000000e+00,   323.199],
       [  0.00000000e+00,   391.05,   241.385],
       [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]]

mtx = np.asarray(mtxe)

diste = [[ 0],
       [  0],
       [  0],
       [ 0],
       [ 0],
       [ 0],
       [ 0],
       [  0],
       [  0.00000000e+00],
       [  0.00000000e+00],
       [  0.00000000e+00],
       [  0.00000000e+00],
       [  0.00000000e+00],
       [  0.00000000e+00]]

dist = np.asarray(diste)

'''
while True:
    ret, frame = cap.read()
    if not ret:
        break
    (corners, ids, rejected) = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    markerSizeInM = 0.176
    rvec , tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerSizeInM, mtx, dist)
    #print(rvec)
    print(tvec)
cap.release()
'''

path = "/home/hanta/hanta_real/240426"

dirs = [f for f in listdir(path)]

for dir in dirs:
    images = [fi for fi in listdir(join(path, dir, "color")) if isfile(join(path, dir, "color", fi))]
    
    images.sort()
    
    if not os.path.exists(join(path, dir, "aruco")):
            os.makedirs(join(path, dir, "aruco"))
    
    
    for image in images:
        frame = cv.imread(join(path, dir, "color", image))
        (corners, ids, rejected) = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        markerSizeInM = 0.176
        rvec , tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerSizeInM, mtx, dist)
        #print(rvec)
        #print(tvec)
        if tvec is not None:
#            result = Project(np.array([[tvec[0, 0, 0],tvec[0, 0, 1],tvec[0, 0, 2]]]), mtx, dist)
            #print(result)
            f = open(join(path, dir, "aruco", image+".txt"), 'w')
            data = "%f\n" % tvec[0, 0, 0]
            f.write(data)
            data = "%f\n" % tvec[0, 0, 1]
            f.write(data)
            data = "%f\n" % tvec[0, 0, 2]
            f.write(data)
            data = "%f\n" % rvec[0, 0, 0]
            f.write(data)
            data = "%f\n" % rvec[0, 0, 1]
            f.write(data)
            data = "%f\n" % rvec[0, 0, 2]
            f.write(data)
            f.close()
        
    
    #print(images)
    
    print(dir)

#print(dirs)
