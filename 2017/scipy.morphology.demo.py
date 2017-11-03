# -*- coding: utf-8 -*-
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
import numpy as np
from scipy import ndimage
a = np.zeros((5,5))
a[2,2] = 1

struct = np.asarray([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]])
print(a)
b = ndimage.binary_dilation(a,struct).astype(a.dtype)
print(b)
 