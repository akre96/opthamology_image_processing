# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:56:07 2020

@author: hwz62
"""


import os
import pandas as pd
import numpy as np
import cv2
from skimage.measure import label
from scipy.spatial import distance


def makeMask(imgpath,yellow=165):
    img=cv2.imread(imgpath)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (yellow-15, 10, 0), (yellow+15, 30, 255))//255
    return mask

def centroidFromMask(label_image, sizeCutoff=20):
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

def generateCentroid(imgpath,yellow,sizeCutoff):
    mask=makeMask(imgpath,yellow)
    label_image = label(mask)
    centroids=centroidFromMask(label_image,sizeCutoff)
    return centroids