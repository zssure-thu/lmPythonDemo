# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:52:13 2017

@author: zssure

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
"""
#######################################
#import datetime
#from skimage import morphology
#from skimage.morphology import square
Note: skimage.morphology may cause an
        internal error,"mkl_intel_thread.dll"
#######################################
"""
import numpy as np
from scipy import ndimage
import os
import json
import sys
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

#创建腐蚀或膨胀的算子
def createMorphologyOperator(left:int,right:int,anterior:int,posterior:int):
    lOptSize = max(left,right,anterior,posterior)
    lOpt = np.zeros((2*lOptSize+1,2*lOptSize+1))
    lOpt[lOptSize,lOptSize] = 1
    for i in range(lOptSize,lOptSize+right):
        lOpt[lOptSize,i] = 1
    for i in range(lOptSize-left,lOptSize):
        lOpt[lOptSize,i] = 1
    for j in range(lOptSize,lOptSize+posterior):
        lOpt[j,lOptSize] = 1
    for j in range(lOptSize-anterior,lOptSize):
        lOpt[j,lOptSize] = 1

    
def runTest():
    inputFile = "d:\\binaryimage.json"
    outputPath = "d:\\kankan"
    dilateOrErosion = 0
    loaddata = loadBinaryDatafromJson(inputFile)
    operator = createMorphologyOperator(1,3,3,3)
    if dilateOrErosion == '0':
        output = ndimage.binary_dilation(loaddata,operator).astype(loaddata.dtype)
    else:
        output = ndimage.binary_erosion(loaddata,operator).astype(loaddata.dtype)
    print(output)
    outFile = open(outputPath+os.sep+'result.json','w')
    #np.savetxt(outputPath+os.sep+'binaryimage2.txt', contours, delimiter=',',fmt='%s')
    outFile.write('[')
    #    for rowdata in filldata:
    #        imgFile.write(str(rowdata.tolist()))
    #        imgFile.write('\n')
    for index in range(output.shape[0]):
        outFile.write(str(output[index].tolist()))
        if index < output.shape[0]-1:
            outFile.write('\n,')
    outFile.write(']')
    outFile.close()
#runTest()

"""
#################################
#Param 1: binary image path , string
#Param 2: result binary image path ,string
#Param 3: dilate(0) or erosion(1)
#Param 4,5,6,7: the operator size , left,right,anterior,posterior 
#################################
"""
if __name__ == "__main__":
    
    inputFile = sys.argv[1]
    outputPath = sys.argv[2]
    dilateOrErosion = sys.argv[3]
    left = sys.argv[4]
    right = sys.argv[5] 
    anterior = sys.argv[6] 
    posterior = sys.argv[7]
    
    loaddata = loadBinaryDatafromJson(inputFile)
    operator = createMorphologyOperator(int(left),int(right),int(anterior),int(posterior))
    if dilateOrErosion == '0':
        output = ndimage.binary_dilation(loaddata,operator).astype(loaddata.dtype)
    else:
        output = ndimage.binary_erosion(loaddata,operator).astype(loaddata.dtype)
    
    outFile = open(outputPath+os.sep+'result.json','w')
    #np.savetxt(outputPath+os.sep+'binaryimage2.txt', contours, delimiter=',',fmt='%s')
    outFile.write('[')
    #    for rowdata in filldata:
    #        imgFile.write(str(rowdata.tolist()))
    #        imgFile.write('\n')
    for index in range(output.shape[0]):
        outFile.write(str(output[index].tolist()))
        if index < output.shape[0]-1:
            outFile.write('\n,')
    outFile.write(']')
    outFile.close()

