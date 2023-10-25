#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#make this excel file and for multiple images
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tkinter import Tk
from tkinter.filedialog import askdirectory
import cv2
import os
import glob
import math


# Input target and predictions
targetDir = ''
#targetDir = askdirectory(title='Select Target Directory') # shows dialog box and return the path
predDir = ''
#predDir = askdirectory(title='Select Prediction Directory') # shows dialog box and return the path


os.chdir(predDir)
predictionFiles = glob.glob('*')
dataFrames = []
for predictionFile in predictionFiles:
    os.chdir(predDir)
    prediction = cv2.imread(predictionFile)
    os.chdir(targetDir)
    target = cv2.imread(predictionFile)
    os.chdir('..')
    
    # Class labels
    #classes = ['Background', 'Body', 'Stalk']
    #classes = ['Background', 'Body', 'Tip']
    classes = ['Background', 'Tip']
    
    # Calculating Intersection over Union (IoU) for each class
    iou_scores = []
    for i, cls in enumerate(classes):
        intersection = np.logical_and(target == i, prediction == i)
        union = np.logical_or(target == i, prediction == i)
        iou_score = np.sum(intersection) / np.sum(union)
        iou_scores.append(iou_score)
    
    # Calculating TP, FP, TN, and FN for each class
    confusion = confusion_matrix(target.flatten(), prediction.flatten())
    tp = np.diag(confusion)  # True Positives
    fp = np.sum(confusion, axis=0) - tp  # False Positives
    fn = np.sum(confusion, axis=1) - tp  # False Negatives
    tn = np.sum(confusion) - (tp + fp + fn)  # True Negatives
    
    # Calculating Precision, Recall, and F1-score for each class
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    dice_coefficient = (2 * tp) / (2 * tp + fp + fn)
    
    # Calculating Pixel Accuracy for each class
    pixel_accuracy = tp / (tp + fp + fn)
    
    # Calculating Segmentation Quality (SQ) for each class
    sq = tp / (tp + fp + fn)
    
    # Calculating Recognition Quality (RQ) for each class
    rq = tp / (tp + fn)
    
    # Calculating Panoptic Quality (PQ): PQ = SQ * RQ (averaged across classes)
    allpq = np.mean(sq * rq)
    pq = sq * rq
    
    # Creating a Pandas DataFrame
   # images= [predictionFile]*3
   # if math.isnan(iou_scores[2]):
   #     images = images[0:2]
   #     classes = classes[0:2]
   #     iou_scores = iou_scores[0:2]
    images= [predictionFile]*2
    if math.isnan(iou_scores[1]):
        images = images[0:1]
        classes = classes[0:1]
        iou_scores = iou_scores[0:1]
        
    #images= [predictionFile]*2

    data = {
        'Image': images,
        'Class': classes,
        'F1-score': f1_score,
        'Precision': precision,
        'Accuracy': pixel_accuracy,
        'Recall': recall,
        'Dice Coefficient':dice_coefficient,
        'Panoptic Quality': pq,
        'IoU': iou_scores,
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn
    }
    
    df = pd.DataFrame(data)
    dataFrames.append(df)

dfs = pd.concat(dataFrames)

#find row indices for background, body, and tip
backgroundInd = np.where(dfs['Class']=='Background')[0]
bodyInd = np.where(dfs['Class']=='Body')[0]
stalkInd= np.where(dfs['Class']=='Stalk')[0]
tipInd = np.where(dfs['Class']=='Tip')[0]
#calculate averages for each column
allClassAverages=np.mean(dfs)
backgroundAverages = np.mean(dfs.iloc[backgroundInd])
bodyAverages = np.mean(dfs.iloc[bodyInd])
stalkAverages = np.mean(dfs.iloc[stalkInd])
tipAverages = np.mean(dfs.iloc[tipInd])


#make a new dataFrame with averages
#avgDF = pd.DataFrame(data=[allClassAverages,backgroundAverages,bodyAverages,stalkAverages])
avgDF = pd.DataFrame(data=[allClassAverages,backgroundAverages,tipAverages])
#avgDF = pd.DataFrame(data=[allClassAverages,backgroundAverages,bodyAverages,tipAverages])

finalDF = pd.concat([avgDF,dfs]).reset_index(drop=True)
#concat the new dataFrame with old
# Exporting the DataFrame to a CSV file
finalDF.to_csv('segmentation_evaluation.csv', index=False)


