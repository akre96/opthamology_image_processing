# -*- coding: utf-8 -*-
"""
Created on Sat May  2 21:29:20 2020

@author: hwz62
"""

import os
from centroidfuncs import generateCentroid
import pandas as pd
import cv2
from matplotlib import pyplot as plt

# File 27 has few cells
# File 36 has more cells

f=27
imgfiles=os.listdir(os.path.join(os.getcwd(),'imgs'))
xyfiles=os.listdir(os.path.join(os.getcwd(),'xy'))
xy=[pd.read_csv(os.path.join(os.getcwd(),'xy',i),index_col=0) for i in xyfiles]
coords=dict(zip(xyfiles,xy))

imgpath=os.path.join(os.getcwd(),'imgs',imgfiles[f])
bestCentroids=generateCentroid(imgpath,20,35)
img=cv2.imread(os.path.join(os.getcwd(),'imgs',imgfiles[f]))
asdf=pd.read_csv(os.path.join(os.getcwd(),'xy',xyfiles[f]),index_col=0)
plt.imshow(img)
plt.scatter(bestCentroids[:,0],bestCentroids[:,1],c='y',s=15)
plt.scatter(asdf['X'],asdf['Y'],c='r',s=15)