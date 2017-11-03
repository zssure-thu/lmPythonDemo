# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:52:13 2017

@author: zssure

Note:
    Firstly, find holes ini binary image or 2D-array , then fill holes by flood-algorithm.
    Secondly, try to find contours in binary image or 2D-array,which be filled.
"""
"""
			  platform : win-64
      conda version : 4.3.30
   conda is private : False
  conda-env version : 4.3.30
conda-build version : not installed
     python version : 3.6.0.final.0
   requests version : 2.18.1
   root environment : C:\Users\Administrator\Anaconda3  (writable)
default environment : C:\Users\Administrator\Anaconda3
   envs directories : C:\Users\Administrator\Anaconda3\envs
                      C:\Users\Administrator\AppData\Local\conda\conda\envs
                      C:\Users\Administrator\.conda\envs
      package cache : C:\Users\Administrator\Anaconda3\pkgs
                      C:\Users\Administrator\AppData\Local\conda\conda\pkgs
       channel URLs : https://conda.anaconda.org/anaconda-fusion/win-64
                      https://conda.anaconda.org/anaconda-fusion/noarch
                      https://repo.continuum.io/pkgs/main/win-64
                      https://repo.continuum.io/pkgs/main/noarch
                      https://repo.continuum.io/pkgs/free/win-64
                      https://repo.continuum.io/pkgs/free/noarch
                      https://repo.continuum.io/pkgs/r/win-64
                      https://repo.continuum.io/pkgs/r/noarch
                      https://repo.continuum.io/pkgs/pro/win-64
                      https://repo.continuum.io/pkgs/pro/noarch
                      https://repo.continuum.io/pkgs/msys2/win-64
                      https://repo.continuum.io/pkgs/msys2/noarch
        config file : C:\Users\Administrator\.condarc
         netrc file : None
       offline mode : False
         user-agent : conda/4.3.30 requests/2.18.1 CPython/3.6.0 Windows/10 Windows/10.0.15063    
      administrator : False
"""
import numpy as np
import datetime
from scipy import ndimage
from skimage import measure
import sys
import os
import json
#导入二值化的二维图像数据
#width图像的宽，代表数组的列数
#height图像的高，代表数组的行数
def loadBinaryDatafromTxt(txtPath,width,height):
    if txtPath and width>0 and height>0:
        data = np.loadtxt(txtPath,delimiter=',')
        if data.ndim ==2 and (data.shape[0] * data.shape[1] == width * height):
            data = np.reshape(data,(height,width))
            return data
        else:
            return []
    else:
        return []
#导入LinkingMed的二维靶区数据
def loadBinaryDatafromJson(jsonPath):
    if jsonPath:
        jsonfile = open(jsonPath)
        jsondata = json.loads(jsonfile.read())
        data = np.asarray(jsondata)
        return data
    else:
        return []
#对二值化的二维图像数组进行孔洞填充，使用Floodfill算法
def floodFillBinaryData(data):
    if len(data.shape)!=2:
        return []
    starttime = datetime.datetime.now()
    img_fill_holes=ndimage.binary_fill_holes(data).astype(int)
    endtime = datetime.datetime.now()
    print(endtime - starttime)
    return img_fill_holes

#对二值化的二维图像数组进行边缘提取，使用MarchingSquare
def marchingSquareBinaryData(data,threshold=0.5):
    starttime = datetime.datetime.now()
    contours = measure.find_contours(data, threshold)
    endtime = datetime.datetime.now()
    print(endtime - starttime)
    return contours

#if __name__ == "__main__":
#    inputFile = sys.argv[1]
#    outputPath = sys.argv[2]
#    width = sys.argv[3]
#    height = sys.argv[4]
#    #loaddata = loadBinaryDatafromTxt(inputFile,int(width),int(height))
#    loaddata = loadBinaryDatafromJson('d:\\1.json',int(514),int(514))
#    filldata = floodFillBinaryData(loaddata)
#    print(filldata)
#    np.savetxt(outputPath+os.sep+'binaryimage.txt', filldata, delimiter=',',fmt='%s')
#    contours = marchingSquareBinaryData(filldata)
#    print(contours)
#    contourFile = open(outputPath+os.sep+'contours.txt','w')
#    #np.savetxt(outputPath+os.sep+'binaryimage2.txt', contours, delimiter=',',fmt='%s')
#    for contour in contours:
#        contourFile.write(str(contour.tolist()))
#        contourFile.write('\n')
#    contourFile.close()
#    

if __name__ == "__main__":
    inputFile = sys.argv[1]
    outputPath = sys.argv[2]
    loaddata = loadBinaryDatafromJson(inputFile)
    filldata = floodFillBinaryData(loaddata)
    #print(filldata)
    #np.savetxt(outputPath+os.sep+'binaryimage.txt', filldata, delimiter=',',fmt='%s')
    imgFile = open(outputPath+os.sep+'binaryimage.json','w')
    #np.savetxt(outputPath+os.sep+'binaryimage2.txt', contours, delimiter=',',fmt='%s')
    imgFile.write('[')
#    for rowdata in filldata:
#        imgFile.write(str(rowdata.tolist()))
#        imgFile.write('\n')
    for index in range(filldata.shape[0]):
        imgFile.write(str(filldata[index].tolist()))
        if index < filldata.shape[0]-1:
            imgFile.write('\n,')
    imgFile.write(']')
    imgFile.close()
    
    contours = marchingSquareBinaryData(filldata)
    #print(contours)
    contourFile = open(outputPath+os.sep+'contours.txt','w')
    #np.savetxt(outputPath+os.sep+'binaryimage2.txt', contours, delimiter=',',fmt='%s')
    for contour in contours:
        contourFile.write(str(contour.tolist()))
        contourFile.write('\n')
    contourFile.close()
    