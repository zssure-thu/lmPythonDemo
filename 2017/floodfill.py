# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:15:35 2017

@author: Administrator
"""

#from scipy import ndimage
#import numpy
#import datetime
import FloodFillAndMarchingSquare as ffms
#data = numpy.loadtxt("D:\\[3]ZSData\\TestData-Floodfill\\floodfill\\test4\\1000w1000h.txt",delimiter=',')
#print(data.shape)
#print(len(data))
#numpy.reshape(data,(1000,1000))
#print(data.shape)
#numpy.savetxt('D:\\[3]ZSData\\TestData-Floodfill\\floodfill\\test4\\output.txt', data, delimiter=',',fmt='%s')
##test 1:
#starttime = datetime.datetime.now()
#img_fill_holes=ndimage.binary_fill_holes(data).astype(int)
#endtime = datetime.datetime.now()
#print(endtime - starttime)
#print(img_fill_holes)
#numpy.savetxt('D:\\[3]ZSData\\TestData-Floodfill\\floodfill\\test4\\output-binaryfill.txt', img_fill_holes, delimiter=',',fmt='%s')
#
##test 2:
#starttime = datetime.datetime.now()
#morph_fill_holes=ndimage.binary_fill_holes(data).astype(int)
#endtime = datetime.datetime.now()
#print(endtime - starttime)
#print(morph_fill_holes)
#numpy.savetxt('D:\\[3]ZSData\\TestData-Floodfill\\floodfill\\test4\\output-morphfill.txt', morph_fill_holes, delimiter=',',fmt='%s')

loaddata = ffms.loadBinaryDatafromTxt("D:\\[3]ZSData\\TestData-Floodfill\\floodfill\\test4\\1000w1000h.txt",1000,1000)
filldata = ffms.floodFillBinaryData(loaddata)
#print(filldata)
contours = ffms.marchingSquareBinaryData(filldata)
print(str(contours[0].tolist()))