# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 20:06:03 2017

@author: zssure
"""

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
