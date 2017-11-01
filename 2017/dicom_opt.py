# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:45:55 2017
@author: zssure
@purpose: examples for dicom operation by Python
"""
#import cv2
import numpy as np
import sys
from PIL import Image
import dicom
from dicom.dataset import Dataset, FileDataset
import datetime, time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from plotly.tools import FigureFactory as FF
#from glob import glob
#from skimage.transform import resize
#from plotly import __version__
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#from plotly.graph_objs import *

#输入加载函数
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    if len(slices) <= 1:
        print ("There is no enough files!")
        sys.exit(1)
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices
#像素获取函数
def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)
#图像显示函数
def show_dicoms(stack,cols=6):
    l = len(stack)
    rows = l // cols
    fig,ax = plt.subplots(rows,cols,figsize=[64,64])
    for i in range(rows*cols-1):
        ax[int(i/cols),int(i % cols)].set_title('slice %d' % i)
        ax[int(i/cols),int(i % cols)].imshow(stack[i],cmap='gray')
        ax[int(i/cols),int(i % cols)].axis('off')
    plt.show()
#图像数据重采样
def resample(image, scan, new_spacing=[1,1,1]):
    '''
    Using the metadata from the DICOM we can figure out the size of each voxel as the slice thickness.
    In order to display the CT in 3D isometric form (which we will do below), and also to compare between 
    different scans, it would be useful to ensure that each slice is resampled in 1x1x1 mm pixels and slices.
    '''
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing
#图像表面提取
def make_mesh(image, threshold=-300, step_size=1):

    p = image.transpose(2,1,0)
    verts, faces = measure.marching_cubes(p, threshold) 
    return verts, faces
#绘制三维表面模型
def plotly_3d(verts, faces):
    x,y,z = zip(*verts) 
    colormap=['rgb(236, 236, 212)','rgb(236, 236, 212)']
    fig = FF.create_trisurf(x=x,
                        y=y, 
                        z=z, 
                        plot_edges=False,
                        colormap=colormap,
                        simplices=faces,
                        backgroundcolor='rgb(64, 64, 64)',
                        title="Interactive Visualization")
    iplot(fig)
#绘制三维模型
def plt_3d(verts, faces):
    x,y,z = zip(*verts) 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    plt.show()
#分割DICOM图像中的肺部
def make_lungmask(img, display=False):
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    #
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation

    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask*img, cmap='gray')
        ax[2, 1].axis('off')
        
        plt.show()
    return mask*img


def write_dicom(pixel_array,index,filename):
    """
    INPUTS:
    pixel_array: 2D numpy ndarray.  If pixel_array is larger than 2D, errors.
    filename: string name for the output file.
    """

    ## This code block was taken from the output of a MATLAB secondary
    ## capture.  I do not know what the long dotted UIDs mean, but
    ## this code works.
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = 'Secondary Capture Image Storage'
    file_meta.MediaStorageSOPInstanceUID = '1.2.3.4.5.6.7.8.9.00'+str(index)
    file_meta.ImplementationClassUID = '1.2.3.4.5.6.7.8.9.0'
    ds = FileDataset(filename, {},file_meta = file_meta,preamble="\0"*128)
    ds.Modality = 'WSD'
    ds.ContentDate = str(datetime.date.today()).replace('-','')
    ds.ContentTime = str(time.time()) #milliseconds since the epoch
    ds.StudyInstanceUID =  '1.2.3.4.5.6.7.8.9'
    ds.SeriesInstanceUID = '1.2.3.4.5.6.7.8.9.00'
    ds.SOPInstanceUID =    file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = 'Secondary Capture Image Storage'
    ds.SecondaryCaptureDeviceManufctur = 'Python 2.7.3 by Linkingmed'

    ## These are the necessary imaging components of the FileDataset object.
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SmallestImagePixelValue = '\\x00\\x00'
    ds.LargestImagePixelValue = '\\xff\\xff'
    ds.Columns = pixel_array.shape[0]
    ds.Rows = pixel_array.shape[1]
    if pixel_array.dtype != np.uint16:
        pixel_array = pixel_array.astype(np.uint16)
    ds.PixelData = pixel_array.tostring()

    ds.save_as(filename)
    return



#if __name__ == "__main__":
##    pixel_array = np.arange(256*256).reshape(256,256)
##    pixel_array = np.tile(np.arange(256).reshape(16,16),(16,16))
#    x = np.arange(16).reshape(16,1)
#    pixel_array = (x + x.T) * 32
#    pixel_array = np.tile(pixel_array,(16,16))
#    write_dicom(pixel_array,'pretty.dcm')
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
=========================Task Examples===============================
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
data_path='D:\\kankan'
'''
1）任务Task示例一：
                task_for_hu，导入DICOM序列，进行像素值转换
2）处理Pipeline：
                load_scan与get_pixels_hu串联
3）算法Algorithm：
                load_scan
                                输入参数：路径（字符串）；
                                输出参数：list
                get_pixels_hu
                                输入参数：list；
                                输出参数：三维数组
'''
patient = load_scan(data_path)
imgs = get_pixels_hu(patient)
'''
1）任务Task示例二：
                task_for_histogram，显示dicom序列的hu灰度统计直方图
2）处理Pipeline：
                load_scan、get_pixels_hu与plt.hist（python自带库）串联
3）算法Algorithm：
                load_scan
                                输入参数：路径（字符串）；
                                输出参数：list
                get_pixels_hu
                                输入参数：list；
                                输出参数：三维数组
                plt.hist
                                输入参数：三维数组
                                输出参数：直方图
                    
'''
#plt.hist(imgs.flatten(), bins=50, color='c')
#plt.xlabel("Hounsfield Units (HU)")
#plt.ylabel("Frequency")
#plt.show()
'''
1）任务Task示例三：
                task_for_sample，对dicom序列数据进行显示
2）处理Pipeline：
                load_scan、get_pixels_hu与show_dicoms串联
3）算法Algorithm：
                load_scan
                                输入参数：路径（字符串）；
                                输出参数：list
                get_pixels_hu
                                输入参数：list；
                                输出参数：三维数组
                show_dicoms
                                输入参数：三维数组、int(每行列数)
                                输出参数：图片
                    
'''
#print ('show_dicoms function starting...')
#show_dicoms(imgs)
'''
1）任务Task示例四：
                task_for_resample，对dicom序列数据进行重采样
2）处理Pipeline：
                load_scan、get_pixels_hu与resample串联
3）算法Algorithm：
                load_scan
                                输入参数：路径（字符串）；
                                输出参数：list
                get_pixels_hu
                                输入参数：list；
                                输出参数：三维数组
                resample
                                输入参数：三维数组、list、一维三元数组
                                输出参数：三维数组、一维三元数组
        
'''
#print ('resample function starting...')
imgs_after_resamp, spacing = resample(imgs, patient, [1,1,1])
#show_dicoms(imgs_after_resamp);
#print ('make_mesh function starting...')
#v, f = make_mesh(imgs_after_resamp,300)
#plt_3d(v, f)

'''
1）任务Task示例五：
                make_lungmasks，分割dicom序列中的肺部
2）处理Pipeline：
                load_scan、get_pixels_hu、resample与make_lungmask串联
3）算法Algorithm：
                load_scan
                                输入参数：路径（字符串）；
                                输出参数：list
                get_pixels_hu
                                输入参数：list；
                                输出参数：三维数组
                resample
                                输入参数：三维数组、list、一维三元数组
                                输出参数：三维数组、一维三元数组
                make_lungmask
                                输入参数：二维数组
                                输出参数：二维数组
'''
#print ('make_lungmask function starting...')
#masked_lung=[]
#for img in imgs_after_resamp:
#    singledcm=make_lungmask(img)
#    masked_lung.append(singledcm)
    
#print ('肺部分割结果展示：')
#show_dicoms(masked_lung)
#print(np.max(singledcm))
masked_lung=[]
index=0
for img in imgs_after_resamp:
    index=index+1
    singledcm=make_lungmask(img)
    plt.imsave('d:\\result-'+str(index)+'.png',singledcm)