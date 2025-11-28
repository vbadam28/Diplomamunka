from logic.pipeline.pipelineFactory import PipelineFactory

import pandas as pd
import nibabel as nib
import numpy as np
import cv2

def loadNifti( path):
    img = nib.load(path)
    return img.get_fdata()



def run():
    #loadImages
    scan  = loadNifti("data_ni/BraTS20_Training_355_flair.nii")
    groundTruthScan = loadNifti("data_ni/W39_1998.09.19_Segm.nii")


    #PrepareImages
    layer = 80
    img = scan[:,:,layer]
    img = cv2.normalize(img,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
    mask = groundTruthScan[:,:,layer]
    mask[mask != 0] = 1


    if np.count_nonzero(mask)<1:
        print("Nincs tumor a rétegen")
        exit()


    #Lehet számolni
    tumor = img[mask!=0]

    #gloabalMean = np.sum(everymeans * everycounts) / np.sum(everycounts)
    avgIntensities =  countIntensities = []
    #avgMinI = np.sum(mins) / countmins     - countmins = images

    prevMinI = 999999
    prevMaxI = avgI = 0

    avgI = np.mean(tumor)
    maxI = np.max(tumor)
    minI = np.min(tumor)
    count = np.sum(mask)

    print(f"Min: {minI}, maxI: {maxI}, avgI: {avgI}, count: {count}")



    from matplotlib import pyplot as plt
    plt.figure()
    plt.subplot(1,2,1)
    displayImg = img.copy()
    displayImg[mask==0]=0
    plt.imshow(displayImg, cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap="gray")
    plt.show()











