# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:56:07 2020

@author: hwz62
"""


import os
import pandas as pd
import numpy as np
import cv2
import SimpleITK as sitk
from skimage.measure import label
from scipy.spatial import distance


def makeMask(imgmat,yellow=165):
    """ Generate mask according to HSV thresholding

    Arguments:
        imgmat: the array of the image to create a threshold mask from
        yellow: saturation value (S in HSV) around which a color range is to be captured 
            as threshold. A +15/-15 boundary is used centered at yellow 
    
    Return: a mask with 1 indicating desired image and 0 in rejected regions. To be used 
        into generateCentroid
    """
    img = sitk.GetImageFromArray(imgmat)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (yellow-15, 10, 0), (yellow+15, 30, 255))//255
    return mask

def centroidFromMask(label_image, sizeCutoff=20):
    """ Identify centroids of cells 

    Arguments:
        imgmat: the array of the image to create a threshold mask from
        yellow: saturation value (S in HSV) around which a color range is to be captured 
            as threshold. A +15/-15 boundary is used centered at yellow 
    
    Return: a mask with 1 indicating desired image and 0 in rejected regions. To be used 
        into generateCentroid
    """
    sizes=[]
    for i in np.unique(label_image):
        sizes.append(sum(sum(label_image==i)))
    sizes=np.array(sizes)
    for i in range(1,len(sizes)):
        if sizes[i]<sizeCutoff:
            posr,posc=np.where(label_image==i)
            for j in range(len(posr)):
                label_image[posr[j],posc[j]]=0
    centroids=[]
    for i in np.unique(label_image):
        posr,posc=np.where(label_image==i)
        centroidr=sum(posr)//len(posr)
        centroidc=sum(posc)//len(posc)
        centroids.append([centroidr,centroidc])
    centroids=np.array(centroids)
    return centroids

def centroidDists(predictedarr,trutharr):
    distances=np.zeros((len(predictedarr),len(trutharr)))
    for i in range(len(predictedarr)):
        for j in range(len(trutharr)):
            pt1=predictedarr[i,:]
            pt2=trutharr[j,:]
            distances[i,j]=distance.euclidean(pt1,pt2)
    return distances

def pipeline(imgpath,yellow=30,sizeCutoff=20): #only used for hyperparameter tuning
    mask=makeMask(imgpath,yellow)
    label_image = label(mask)
    centroids=centroidFromMask(label_image,sizeCutoff)
    groundtruth=pd.read_csv(os.path.join(os.getcwd(),'xy',xyfiles[f]),index_col=0)
    groundtruth=np.array(groundtruth)
    distances=centroidDists(centroids,groundtruth)
    sumError=sum(np.amin(distances,1))+sum(np.amin(distances,0))
    avgError=sumError/(len(centroids)+len(groundtruth))
    return avgError

def generateCentroid(imgmat,yellow,sizeCutoff):
    mask=makeMask(imgmat,yellow)
    label_image = label(mask)
    centroids=centroidFromMask(label_image,sizeCutoff)
    return centroids
