# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 16:46:49 2020

@author: hwz62
"""
import load_data as ld
import segment_tiles as st
from radfuncs import radfeatures
import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import linregress
import SimpleITK as sitk
import json
import sampledata as data


def createFeatures(label):
    """ Generates underlying features to be used for cell density prediction

    Arguments:
        label -- the cass that the image upon which features are being
            generated is classified under, eg. disease severity. If not known,
            enter "0".

    Returns:
        features -- calculated texture features. Also saved as file "features.csv"
            in current directory.
    """
    tile_select_params = ld.get_tile_select_params()['BEC']
    segmenter = st.TileSegmenter(
        **tile_select_params,
        plot_patches=False,
        show_tqdm=False,
    )

    metadata = ld.load_bcd_metadata()

    def generateImg(index, metadata):
        example_sub = metadata.iloc[index]
        example_scan = example_sub['cc scan 1']
        patient_sample = example_sub['study_number_id_eye']
        img_class = example_sub['clinical']
        pid, img_slice = ld.convert_id_to_folder_label(
            patient_sample,
            example_scan
        )
        img = ld.get_bcd_image(img_class, pid, img_slice, cv2.IMREAD_UNCHANGED)
        patches = segmenter.segment_tiles(img)
        return patches

    index = metadata.shape[0]
    patches_all = []
    for i in range(index):
        patches_all.append(generateImg(i, metadata))

    patches_all = [item for sublist in patches_all for item in sublist]

    # Pull texture features
    features = pd.DataFrame()
    for i, img in enumerate(patches_all):
        features[i] = radfeatures(img, label)

    features = features.transpose()
    features.columns = ['original_firstorder_10Percentile', 'original_firstorder_90Percentile', 'original_firstorder_Energy', 'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis', 'original_firstorder_Maximum', 'original_firstorder_MeanAbsoluteDeviation', 'original_firstorder_Mean', 'original_firstorder_Median', 'original_firstorder_Minimum', 'original_firstorder_Range', 'original_firstorder_RobustMeanAbsoluteDeviation', 'original_firstorder_RootMeanSquared', 'original_firstorder_Skewness', 'original_firstorder_TotalEnergy', 'original_firstorder_Uniformity', 'original_firstorder_Variance', 'original_glcm_Autocorrelation', 'original_glcm_ClusterProminence', 'original_glcm_ClusterShade', 'original_glcm_ClusterTendency', 'original_glcm_Contrast', 'original_glcm_Correlation', 'original_glcm_DifferenceAverage', 'original_glcm_DifferenceEntropy', 'original_glcm_DifferenceVariance', 'original_glcm_Id', 'original_glcm_Idm', 'original_glcm_Idmn', 'original_glcm_Idn', 'original_glcm_Imc1', 'original_glcm_Imc2', 'original_glcm_InverseVariance', 'original_glcm_JointAverage', 'original_glcm_JointEnergy', 'original_glcm_JointEntropy', 'original_glcm_MCC', 'original_glcm_MaximumProbability', 'original_glcm_SumAverage', 'original_glcm_SumEntropy', 'original_glcm_SumSquares', 'original_gldm_DependenceEntropy', 'original_gldm_DependenceNonUniformity', 'original_gldm_DependenceNonUniformityNormalized', 'original_gldm_DependenceVariance', 'original_gldm_GrayLevelNonUniformity', 'original_gldm_GrayLevelVariance', 'original_gldm_HighGrayLevelEmphasis', 'original_gldm_LargeDependenceEmphasis', 'original_gldm_LargeDependenceHighGrayLevelEmphasis', 'original_gldm_LargeDependenceLowGrayLevelEmphasis', 'original_gldm_LowGrayLevelEmphasis', 'original_gldm_SmallDependenceEmphasis', 'original_gldm_SmallDependenceHighGrayLevelEmphasis', 'original_gldm_SmallDependenceLowGrayLevelEmphasis', 'original_glrlm_GrayLevelNonUniformity', 'original_glrlm_GrayLevelNonUniformityNormalized', 'original_glrlm_GrayLevelVariance', 'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 'original_glrlm_LongRunHighGrayLevelEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis', 'original_glrlm_LowGrayLevelRunEmphasis', 'original_glrlm_RunEntropy', 'original_glrlm_RunLengthNonUniformity', 'original_glrlm_RunLengthNonUniformityNormalized', 'original_glrlm_RunPercentage', 'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 'original_glrlm_ShortRunLowGrayLevelEmphasis', 'original_glszm_GrayLevelNonUniformity', 'original_glszm_GrayLevelNonUniformityNormalized', 'original_glszm_GrayLevelVariance', 'original_glszm_HighGrayLevelZoneEmphasis', 'original_glszm_LargeAreaEmphasis', 'original_glszm_LargeAreaHighGrayLevelEmphasis', 'original_glszm_LargeAreaLowGrayLevelEmphasis', 'original_glszm_LowGrayLevelZoneEmphasis', 'original_glszm_SizeZoneNonUniformity', 'original_glszm_SizeZoneNonUniformityNormalized', 'original_glszm_SmallAreaEmphasis', 'original_glszm_SmallAreaHighGrayLevelEmphasis', 'original_glszm_SmallAreaLowGrayLevelEmphasis', 'original_glszm_ZoneEntropy', 'original_glszm_ZonePercentage', 'original_glszm_ZoneVariance', 'original_ngtdm_Busyness', 'original_ngtdm_Coarseness', 'original_ngtdm_Complexity', 'original_ngtdm_Contrast', 'original_ngtdm_Strength','label']
    features.to_csv('features.csv')
    return features



# Load features then run prediction


def trainModel(features, densities_true):
    """ Trains linear regression model on PCA-transformed basal cell images 
    texture features to calculate parameters

    Arguments:
        features -- texture features identified using the 'createFeatures'
            function
        densities_true -- known densities from prior experimentation
            or hand-labeling

    Returns:
        slope, intercept -- the slope and intercept for a linear regression
            model performed from texture features against known densities.
            Saves a JSON file of statistics.
            Saves a JSON of PCA components
    """
    Y = list(features['label'])
    X = features.drop(['label'], axis=1)
    X_norm = StandardScaler().fit_transform(X)
    pca = PCA(n_components=10)
    pca_result = pca.fit(X_norm)
    X_pca = pca_result.transform(X_norm)
    slope, intercept, r_value, p_value, std_err = linregress(X_pca[:, 0], densities_true)
    jsonout = {'slope': slope,
               'intercept': intercept,
               'r_value': r_value,
               'p-value': p_value}
    with open('regressiondata.json', 'w') as json_file:
        json.dump(jsonout, json_file)
    np.savetxt("pcaparams.csv", pca_result.components_, delimiter=",")
    return slope, intercept


def predictDensity(features):
    with open('regressiondata.json') as f:
        reg=json.load(f)
    Y = list(features['label'])
    X = features.drop(['label'], axis=1)
    X_norm = StandardScaler().fit_transform(X)
    eigenvectors = np.loadtxt(open("pcaparams.csv", "rb"), delimiter=",")
    features_tf = X@eigenvectors[0,:]
    pred_density=reg['slope']*features_tf+reg['intercept']
    return pred_density


def pipe(label):
    features = createFeatures(label)
    predictions=predictDensity(features)
    return predictions

#test=pipe('test')
