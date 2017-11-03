# -*- coding: utf-8 -*-
#"""
#Created on Tue Oct 31 20:06:03 2017
#
#@author: zssure
#"""
#"""
#			  platform : win-64
#      conda version : 4.3.30
#   conda is private : False
#  conda-env version : 4.3.30
#conda-build version : not installed
#     python version : 3.6.0.final.0
#   requests version : 2.18.1
#   root environment : C:\Users\Administrator\Anaconda3  (writable)
#default environment : C:\Users\Administrator\Anaconda3
#   envs directories : C:\Users\Administrator\Anaconda3\envs
#                      C:\Users\Administrator\AppData\Local\conda\conda\envs
#                      C:\Users\Administrator\.conda\envs
#      package cache : C:\Users\Administrator\Anaconda3\pkgs
#                      C:\Users\Administrator\AppData\Local\conda\conda\pkgs
#       channel URLs : https://conda.anaconda.org/anaconda-fusion/win-64
#                      https://conda.anaconda.org/anaconda-fusion/noarch
#                      https://repo.continuum.io/pkgs/main/win-64
#                      https://repo.continuum.io/pkgs/main/noarch
#                      https://repo.continuum.io/pkgs/free/win-64
#                      https://repo.continuum.io/pkgs/free/noarch
#                      https://repo.continuum.io/pkgs/r/win-64
#                      https://repo.continuum.io/pkgs/r/noarch
#                      https://repo.continuum.io/pkgs/pro/win-64
#                      https://repo.continuum.io/pkgs/pro/noarch
#                      https://repo.continuum.io/pkgs/msys2/win-64
#                      https://repo.continuum.io/pkgs/msys2/noarch
#        config file : C:\Users\Administrator\.condarc
#         netrc file : None
#       offline mode : False
#         user-agent : conda/4.3.30 requests/2.18.1 CPython/3.6.0 Windows/10 Windows/10.0.15063    
#      administrator : False
#"""
#Marching Square Demo

import matplotlib.pyplot as plt
from skimage import measure
import datetime
import numpy as np
starttime = datetime.datetime.now()
data = np.loadtxt("D:\\[3]ZSData\\TestData-Floodfill\\floodfill\\test4\\output-binaryfill.txt",delimiter=',')
np.reshape(data,(1000,1000))
endtime = datetime.datetime.now()
print(endtime - starttime)
## Construct some test data
#x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
#r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))

# Find contours at a constant value of 0.8
contours = measure.find_contours(data, 0.5)
print(contours)
#np.savetxt('D:\\[3]ZSData\\TestData-Floodfill\\floodfill\\test4\\output-contours.txt', contours[1], delimiter=',',fmt='%s')
# Display the image and plot all contours found
fig, ax = plt.subplots()

#ax.imshow(data, interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=1)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
