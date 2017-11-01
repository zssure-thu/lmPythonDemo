#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:16:21 2017

@author: sy
"""

from keras import models
from glob import glob
from matplotlib import pyplot as plt
import nibabel as nb
import numpy as np
import datetime
import os
import SimpleITK as sitk
import scipy.ndimage as nd
import dicom
import cv2
import interpolation
from copy import deepcopy
from skimage.measure import label as ll
from skimage.measure import regionprops
from skimage import filters,segmentation,measure,morphology,color
from scipy import interpolate  


os.environ['KERAS_BACKEND'] = 'tensorflow'  
#os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN, device=gpu, floatX=float32, optimizer=fast_compile,nvcc.flags=-D_FORCE_INLINES'
os.environ['TENSORFLOW_FLAGS'] = 'mode=FAST_RUN, device=gpu, floatX=float32, \
                                    optimizer=fast_compile,nvcc.flags=-D_FORCE_INLINES'
# ==============================================================================
# 1.导入dicom文件，并进行插值,输出的数据格式是z, x, y  
# 输入是dicom所在路径
# example:    
# path = r'E:\data_original\csv\129724_225800_1\1.3.6.1.4.1.2452.6.2085389675.1192291963.3012834467.897554807\CT'
# data = load_data(path)
# ==============================================================================
def load_data(data_path):

    slices = [dicom.read_file(data_path + os.sep + s) for s in os.listdir(data_path)]    
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]), reverse=False)

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    spacing = slices[1].PixelSpacing    
    spacing = map(float, ([slices[2].SliceThickness] + spacing))
    spacing = np.array(list(spacing))

    spacing[0], spacing[2] = spacing[2], spacing[0]

    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    image = image.astype(np.float32)
    
    return image, spacing[:1]
    
# ==============================================================================
# 2.插值函数，线性插值， 该函数在第一个函数中调用
# 输入：原图； 之前的spcing； 希望得到的spcing
# example:    
# image = nearest_interpolation(image, spacing[:1], [1.5, 1.5])
# ==============================================================================
def nearest_interpolation(data, spacing, std, pro):

    print ('start interpolation......')
    if pro == 'space':
        pixel = np.array(spacing, dtype = np.float32) / np.array(std, dtype = np.float32) 
        z,x,y = np.shape(data)
        xn = int(x*pixel[0])
        yn = int(y*pixel[1])
    
        img_new3 = np.zeros([z, xn, yn], dtype = np.float32)
        for i in range(z):
            img = data[i,:,:]
            
            xx = np.linspace(-1,1,x)
            yy = np.linspace(-1,1,y)
     
            xxn = np.linspace(-1,1,xn)
            yyn = np.linspace(-1,1,yn)
            
            newfunc = interpolate.interp2d(xx, yy, img, kind='linear')  
            fnew = newfunc(xxn, yyn)
            img_new3[i,:,:] = fnew
                
        return img_new3
    if pro == 'len':
        z, x,y = np.shape(data)
        xn = int(std[0])
        yn = int(std[1])
    
        img_new3 = np.zeros([z, xn, yn], dtype = np.float32)
        for i in range(z):
            img = data[i,:,:]
            
            xx = np.linspace(-1,1,x)
            yy = np.linspace(-1,1,y)
     
            xxn = np.linspace(-1,1,xn)
            yyn = np.linspace(-1,1,yn)
            
            newfunc = interpolate.interp2d(xx, yy, img, kind='linear')  
            fnew = newfunc(xxn, yyn)
            img_new3[i,:,:] = fnew
                
        return img_new3
        
# ==============================================================================
# 3.图像裁剪函数，输入三维图像，将其裁剪为我们理想的大小。
#   同时也是图像填充函数                                   
# 输入：原图
# example:    
# image_dataset = std_data(data, 150)
# ==============================================================================
def std_data(data, image_size, pro):
    if pro == 'crop':
        z, x, y = np.shape(data)
        judge = sum([x > image_size, y > image_size])
        
        if judge == 2 :
            image_std = data[:, int((x-image_size)/2):int((x+image_size)/2), int((y-image_size)/2):int((y+image_size)/2)]
        if judge == 0:
            image_new = -np.min(data)*np.ones([z,image_size, image_size], dtype = np.int32)
            image_new[:, int((image_size-x)/2):int((image_size-x)/2)+x, int((image_size-y)/2):int((image_size-y)/2)+y] = data
            image_std = image_new
        if judge == 1:
            ValueError('输入图像的长宽不一致，请调节或重新编辑')
    
        image_std = np.reshape(image_std, [z, image_size, image_size, 1])
        
        return image_std   
    elif pro == 'pad':
        z, x, y = np.shape(data)
        judge = sum([x < image_size, y < image_size])
        
        if judge == 2 :
            image_std = np.zeros([z,image_size,image_size])
            image_std[:, int((image_size-x)/2):int((image_size+x)/2), int((image_size-y)/2):int((image_size+y)/2)] = data
        if judge == 0:
            image_std = data[:, int((x-image_size)/2):int((x+image_size)/2)+x, int((y-image_size)/2):int((y+image_size)/2)+y]
        if judge == 1:
            ValueError('输入图像的长宽不一致，请调节或重新编辑')
          
        return image_std   
                                    
# ==============================================================================
# 4.分类网络
# ==============================================================================
def calss_organ(data, modle_path):
    
    image_dataset = std_data(data, 150, 'crop')

    with open(modle_path + 'stem_spinalcord_net.json') as file:
        my_model = models.model_from_json(file.read())
    my_model.load_weights(modle_path + 'stem_spinalcord_net.h5')
    
    result = my_model.predict(image_dataset, batch_size=32, verbose=1)
    
    result = np.argmax(result, axis=1)
    ind_2 = [i for i,a in enumerate(result) if a==2]
    ind_1 = [i for i,a in enumerate(result) if a==1]
    try:
        length = int(len(ind_1)/6)
        ind = ind_1[length:-length]
        ind = np.array(ind)
        ind_final_1 = list(range(ind.min()-length, ind.max()+length+1))
        
        length = int(len(ind_2)/6)
        ind = ind_2[length:-length]
        ind = np.array(ind)
        ind_final_2 = list(range(ind.min()-length, ind.max()+length+1))
    except:
        ind_final_1 = ind_1
        ind_final_2 = ind_2
   
    return ind_final_2, ind_final_1
    
# ==============================================================================
# 5.脊髓分割网络
# ==============================================================================
def seg_spinalcord(data, modle_path):
    
    image_dataset = std_data(data, 256, 'crop')

    with open(modle_path + 'spinalcord.json') as file:
        my_model = models.model_from_json(file.read())
    my_model.load_weights(modle_path + 'spinalcord.h5')
    
    result = my_model.predict(image_dataset, batch_size=32, verbose=1)
    
    return result

# ==============================================================================
# 6.脊髓分割后处理
# ==============================================================================
def std_spinalcord(spinalcord_label):
    
    rate = 0.4
    for ind, i in enumerate(spinalcord_label):   #每张图象阈值处理
        spinalcord_label[ind,:,:,0] = i[:,:,0] > (np.max(i[:,:,0]) * rate)
    
    spinalcord_label = np.array(spinalcord_label, dtype = np.int16)
    label = deepcopy(spinalcord_label)
    if np.sum(label[0]) < np.sum(label[1]) * 0.5:  #防止第一层分割错误
        label[0] = label[1]
    
    for ind, label_slice in enumerate(label):
        label_slice = label_slice[:,:,0]
        if ind > 0 and np.sum(label_slice) == 0:
            label_slice = label[ind - 1,:,:,0]
        label_num = ll(label_slice, connectivity  = 2)   # 连通区域编号
        label_pro = regionprops(label_num)    # 每个连通区域的属性
        x, y = np.shape(label_num)
    
        if len(label_pro) > 1 and ind > 1:
            label_slice_1 = label[ind - 1,:,:,0] 
            label_num_1 = ll(label_slice_1, connectivity  = 2)   # 连通区域编号
            label_pro_1 = regionprops(label_num_1)
            label_pro_1_cen = np.array(label_pro_1[0].centroid, dtype = np.float16)
            cen_len = []
            for i in range(len(label_pro)):
                label_pro_cen = np.array(label_pro[i].centroid, dtype = np.float16)
                length = np.sqrt(np.sum(abs(label_pro_1_cen - label_pro_cen) * abs(label_pro_1_cen - label_pro_cen)))
                cen_len.append(length)
                
            index = np.argmin(cen_len)
            label_slice = np.array(label_num == index + 1, dtype = np.int16)
            
        if ind > 1:

            label_num = ll(label_slice, connectivity  = 2)   # 连通区域编号
            label_pro = regionprops(label_num)    # 每个连通区域的属性
            label_pro_area = label_pro[0].area
            
            label_slice_1 = label[ind - 1,:,:,0] 
            label_num_1 = ll(label_slice_1, connectivity  = 2)   # 连通区域编号
            label_pro_1 = regionprops(label_num_1)
            label_pro_1_area = label_pro_1[0].area
    
            if label_pro_area < label_pro_1_area * 0.5:
                label_slice = label[ind-1,:,:,0]
        label[ind,:,:,0] = label_slice  
    label = np.array(label > 0, np.uint8)
    return label

# ==============================================================================
# 5.脑干分割网络
# ==============================================================================
def seg_brainstem(data, modle_path):
    
    image_dataset = std_data(data, 256, 'crop')

    with open(modle_path + 'brainstem.json') as file:
        my_model = models.model_from_json(file.read())
    my_model.load_weights(modle_path + 'brainstem.h5')
    
    result = my_model.predict(image_dataset, batch_size=32, verbose=1)
    return result

    
# ==============================================================================
# 8.脑干分割后处理
# ==============================================================================
def std_brainstem(spinalcord_label):     
    rate = 0.4
    for ind, i in enumerate(spinalcord_label):   #每张图象阈值处理
        spinalcord_label[ind,:,:,0] = i[:,:,0] > (np.max(i[:,:,0]) * rate)
    
    spinalcord_label = np.array(spinalcord_label, dtype = np.int16)
    label = deepcopy(spinalcord_label)
    if np.sum(label[0]) < np.sum(label[1]) * 0.5:  #防止第一层分割错误
        label[0] = label[1]
    
    for ind, label_slice in enumerate(label):
        label_slice = label_slice[:,:,0]
        if ind > 0 and np.sum(label_slice) == 0:
            label_slice = label[ind - 1,:,:,0]
        label_num = ll(label_slice, connectivity  = 2)   # 连通区域编号
        label_pro = regionprops(label_num)    # 每个连通区域的属性
        x, y = np.shape(label_num)

        if len(label_pro) > 1 and ind > 1:
            judge = [x.area for x in label_pro]
            label_slice = morphology.remove_small_objects(label_num,min_size=np.max(judge),connectivity=1)
            
        if ind > 1:
            
            label_num = ll(label_slice, connectivity  = 2)   # 连通区域编号
            label_pro = regionprops(label_num)    # 每个连通区域的属性
            label_pro_area = label_pro[0].area
            label_pro_cen = np.array(label_pro[0].centroid, dtype = np.float16)
            
            label_slice_1 = label[ind - 1,:,:,0] 
            label_num_1 = ll(label_slice_1, connectivity  = 2)   # 连通区域编号
            label_pro_1 = regionprops(label_num_1)
            label_pro_1_area = label_pro_1[0].area
            label_pro_1_cen = np.array(label_pro_1[0].centroid, dtype = np.float16)
   
            length = np.sqrt(np.sum(abs(label_pro_1_cen - label_pro_cen) * abs(label_pro_1_cen - label_pro_cen)))
    
            if label_pro_area < label_pro_1_area * 0.9 or length > np.sqrt(label_pro_1_area) / 2 :
                label_slice = label[ind-1,:,:,0]
            label[ind,:,:,0] = label_slice 
    label = np.array(label > 0, np.uint8)

    for ind, label_slice in enumerate(label):
        label_slice = label_slice[:,:,0]
        if np.sum(label_slice) > 250:
            kernel = np.ones((5,5),np.uint8)
            label_slice = cv2.morphologyEx(label_slice,cv2.MORPH_OPEN, kernel)
        label_num = ll(label_slice, connectivity  = 2)   # 连通区域编号
        label_pro = regionprops(label_num)    # 每个连通区域的属性
        if len(label_pro) > 1 and ind > 1:
            judge = [x.area for x in label_pro]
            label_slice = morphology.remove_small_objects(label_num,min_size=np.max(judge),connectivity=1)
        label[ind,:,:,0] = label_slice 
    label = np.array(label > 0, np.uint8)
    return label    
    
if __name__ == '__main__':
    time1 = datetime.datetime.now()
    modle_path = r'.' + os.sep  #模型所在路径
    #path = r'E:\data_original\csv\129724_225800_1\1.3.6.1.4.1.2452.6.2085389675.1192291963.3012834467.897554807\CT'
    path = r'D:\[3]ZSData\TestData-CT\chen san de_NPC\CT'
    # =========================================================================
    # 1.数据导入
    # z, x, y   
    data, spacing = load_data(path)  #未插值之前
    z0, x0, y0 = np.shape(data)      #未插值之前
    data = np.array(data, dtype = np.int16)
    data = nearest_interpolation(data, spacing, [1.5, 1.5], 'space')#插值之后
    z, x, y = np.shape(data)         #插值之后
    print(z, x, y)
    # =========================================================================
    # 2.脑干, 脊髓分类
    # z, x, y    120, 333, 333
    ind_spinal, ind_stem = calss_organ(data, modle_path)    # 脑干和脊髓的z轴坐标分布
    ind_spinal = list(range(ind_spinal[0], ind_stem[0]))    # 防止脑干和脊髓中间有空白层
    spinalcord_dataset = data[ind_spinal[0]:ind_spinal[-1]+1, :, :] # 提取脊髓数据集
    brainstem_dataset = data[ind_stem[0]:ind_stem[-1]+1, :, :] # 提取脑干数据集
    # =========================================================================
    # 3.脊髓分割
    label_1 = seg_spinalcord(spinalcord_dataset, modle_path)    # 得到初步分割的结果
    label_spinal = np.zeros([z, 256, 256, 1])      # 在z轴上将label回填回去，先建立0集
    label_spinal[ind_spinal[0]:ind_spinal[-1]+1, :, :, :] = std_spinalcord(label_1)   
    label_spinal = std_data(np.reshape(label_spinal, [z,256,256]), x, 'pad')  # 在x， y平面将label填回去
    label_spinal = np.array(label_spinal, dtype = np.int16)
    label_spinal = nearest_interpolation(label_spinal, [x, x], [x0, x0], 'len')
    # z, x, y
    # =========================================================================
    # 4.脑干分割
    label_2 = seg_brainstem(brainstem_dataset, modle_path)# 得到初步分割的结果
    label_stem = np.zeros([z, 256, 256, 1])      # 在z轴上将label回填回去，先建立0集
    label_stem[ind_stem[0]:ind_stem[-1]+1, :, :, :] = std_spinalcord(label_2)
    label_stem = np.reshape(label_stem, [z,256,256])
    label_stem = std_data(np.reshape(label_stem, [z,256,256]), x, 'pad')  # 在x， y平面将label填回去
    label_stem = np.array(label_stem, dtype = np.int16)
    label_stem = nearest_interpolation(label_stem, [x, x], [x0, x0], 'len')
    # z, x, y
    # =========================================================================
    time2 = datetime.datetime.now()
    print(time2 - time1)
    
    # =========================================================================
    bb = nb.Nifti1Image(label_spinal, affine=np.eye(4))
    nb.save(bb, r'b.nii.gz')
    cc = nb.Nifti1Image(label_stem, affine=np.eye(4))
    nb.save(cc, r'a.nii.gz')
    
    