#!/usr/bin/env python
from __future__ import print_function


# ## filoTips Instructions:
# 
# 1.   Collect images of your cells making filopodia. Please provide one channel: fluorescent tip-enriched protein that robustly labels cell body and filopodia tips. Z-stack max projection recommended.
# 1.   Click "Cell" in the toolbar above, then "Run All"
# 1.   All cell headers (In [ ]:) should now contain an asterisk within the brackets, noting they are running either now or in the future. As the cells complete their runs, the asterisk in the cell headers (In [ * ]:) will be replaced in the order in which they ran (In [1]:, In [2]:, etc.). If a cell has completed a run then it should contain a number.
# 1.   Please provide the input asked of you in cell 1.1. The first input will ask you to select the folder containing the images you want to analyze
# 2.   If Custom model is selected, please provide: **a)** the Google Drive link to a **compressed (.zip) folder containing your model** (ex: "*1zh5j_VL380ebyQxDCJ5N2fhOxWqkFQsC*") and **b)** the model/zip file name ("*Stanley*"). This can be found by right clicking your .zip model, clicking "Share", then "Anyone with link", and finally "Copy Link". Paste it here, then remove all but the specific link which should look similar to the example link provided above.
# 1.   Wait for all cells to run (From "In [*]:" to "In [num]:) and look for the "-- filoTips analysis complete --" prompt below cell 1.9. This indicates the analysis was a success.

# ### Please provide all required user inputs below:
# 
# Micron/Pixel Ratio: Ratio of microns per pixel (commonly found in image metadata or can be found in ImageJ)\
# \
# Model Type: "Default" filoTips model, or "Custom" model that has been fine-tuned to user data\
# \
# filoSpace: Would you like inter-filopodial spacing information (Note: high image resolution recommended, experimental)\
# \
# Brief statistical summary: Would you like a limited statistical summary (ex. t-test, one-way anova, violin plots)? Must have 2-4 variables or experimental conditions to compare and each image file name must contain a unique string to assign to a specific group. That unique string must be provided for each variable/condition.\
# \
# Figure DPI: Please provide figure DPI. This will impact filoTips annotation resolution and can be adjusted accordingly. Recommended start value is "300". 

# In[1]:


#get filename and micron/pixel ratio
from tkinter import Tk
from tkinter.filedialog import askdirectory
path = askdirectory(title='Select Folder') # shows dialog box and return the path
print(path+' selected as file path')

print('')
um_per_pixel = None
while type(um_per_pixel) != float:
    try:
        um_per_pixel = float(input('Micron/Pixel Ratio: '))
        break
    except ValueError:
        print("Please enter a valid number...")

#what model should be used?
model_type = None
print('')
while model_type not in {"Default", "Custom"}:
    model_type = input('Model type, "Default" or "Custom": ')
    if model_type not in {"Default", "Custom"}:
        print('Please type "Default" or "Custom" verbatim to note which model you want to use.')
if model_type == 'Custom':
    custom_model_DriveLink= input('Please copy and paste the Google DriveLink associated with your custom model: ')
    temp = custom_model_DriveLink.replace('https://drive.google.com/file/d/','')
    temp = temp.replace('/view?usp=drive_link','')
    temp = temp.replace('/view?usp=sharing','')
    custom_model_DriveLink = temp

#do you want interfilopodial spacing information?    
filoSpace = None
print('')
while filoSpace not in {True,False}:
    filoSpace = input('Would you like filoSpace enabled?, "True" or "False": ')
    if filoSpace == 'True':
        filoSpace = True
    if filoSpace == 'False':
        filoSpace = False
    if filoSpace not in {True,False}:
        print('Please type "True" or "False" to note if you would like filoSpace enabled.')
        
# do you want to perform a comparative_analysis        
comparative_analysis = None
print('')
while comparative_analysis not in {True,False}:
    comparative_analysis = input('Would you like a brief statistical summary at the end? More than one experimental condition/variable required and a unique string for each condition/variable must be present in all image file names, "True" or "False": ')
    if comparative_analysis == 'True':
        comparative_analysis = True
        Condition_1 = input('Unique string in file name associated with condition 1: ')
        Condition_2 = input('Unique string in file name associated with condition 2: ')
        Condition_3 = input('Unique string in file name associated with condition 3, leave empty if only testing two conditions: ')
        Condition_4 = input('Unique string in file name associated with condition 4, leave empty if only testing two or three conditions: ')
    if comparative_analysis == 'False':
        comparative_analysis = False

#Define desired figure resolution
Annotation_DPI = None
print('')
while type(Annotation_DPI) != int:
    try:
        Annotation_DPI = int(input('Figure DPI (ex. "150","300","900"): '))
        break
    except ValueError:
        print("Please enter a valid number...")

#Calculate pixel_micron from micron_pixel
pixel_micron=1/float(um_per_pixel)

print('')
print('Inputs accepted. Initiating filoTips analysis...')
print('')

# ### Load requirements from ZeroCostDL4Mic (requirements from "U-Net_2D_Multilabel" notebook)

# In[2]:

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
#code from ZeroCostDL4Mic 1.1
install('pandas')
install('data')
install('fpdf')
install('tensorflow')
install('scikit-image')
install('matplotlib')
install('scikit-learn')
install('ipywidgets')
install('tqdm')
install('seaborn')
install('statsmodels')
subprocess.run('pip install h5py==2.10',shell=True)
#install("h5py==2.10")
install('imagecodecs')

#code from ZeroCostDL4Mic 1.3
Notebook_version = '1.13'
Network = 'U-Net (2D) multilabel'

import imagecodecs
from builtins import any as b_any

def get_requirements_path():
    # Store requirements file in 'contents' directory 
    #current_dir = os.getcwd()
    #dir_count = current_dir.count('/') - 1
    #path = './' * (dir_count) + 'requirements.txt'
    req_path = path+'/requirements.txt'
    return req_path

def filter_files(file_list, filter_list):
    filtered_list = []
    for fname in file_list:
        if b_any(fname.split('==')[0] in s for s in filter_list):
            filtered_list.append(fname)
    return filtered_list

def build_requirements_file(before, after):
    req_path = get_requirements_path()

    # Exporting requirements.txt for local run
    os.chdir(path)
    subprocess.run('pip freeze > requirements.txt',shell=True)

    # Get minimum requirements file
    #df = pd.read_csv(path, delimiter = "\n")
    df = pd.read_csv('requirements.txt')
    mod_list = [m.split('.')[0] for m in after if not m in before]
    req_list_temp = df.values.tolist()
    req_list = [x[0] for x in req_list_temp]

    # Replace with package name and handle cases where import name is different to module name
    mod_name_list = [['sklearn', 'scikit-learn'], ['skimage', 'scikit-image']]
    mod_replace_list = [[x[1] for x in mod_name_list] if s in [x[0] for x in mod_name_list] else s for s in mod_list] 
    filtered_list = filter_files(req_list, mod_replace_list)

    file=open(req_path,'w')
    for item in filtered_list:
        file.writelines(item + '\n')

    file.close()

import sys
before = [str(m) for m in sys.modules]

#As this notebokk depends mostly on keras which runs a tensorflow backend (which in turn is pre-installed in colab)
#only the data library needs to be additionally installed.
#%tensorflow_version 1.x
import tensorflow as tf
# print(tensorflow.__version__)
# print("Tensorflow enabled.")


# Keras imports
from keras import models
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger # we currently don't use any other callbacks from ModelCheckpoints
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import img_to_array
#from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import backend as keras
from keras.callbacks import Callback

# General import
import numpy as np
import pandas as pd
import os
import glob
from skimage import img_as_ubyte, io, transform
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread
from pathlib import Path
import shutil
import random
import time
import csv
import sys
from math import ceil
from fpdf import FPDF, HTMLMixin
from pip._internal.operations.freeze import freeze
import subprocess
# Imports for QC
from PIL import Image
from scipy import signal
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from skimage.util import img_as_uint
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio as psnr

# For sliders and dropdown menu and progress bar
from ipywidgets import interact
import ipywidgets as widgets
# from tqdm import tqdm
from tqdm.notebook import tqdm

from sklearn.feature_extraction import image
from skimage import img_as_ubyte, io, transform
from skimage.util.shape import view_as_windows

from datetime import datetime


# Suppressing some warnings
import warnings
warnings.filterwarnings('ignore')




def create_patches(Training_source, Training_target, patch_width, patch_height, min_fraction):
  """
  Function creates patches from the Training_source and Training_target images. 
  The steps parameter indicates the offset between patches and, if integer, is the same in x and y.
  Saves all created patches in two new directories in the /content folder.

  Returns: - Two paths to where the patches are now saved
  """
  DEBUG = False

  Patch_source = os.path.join('/content','img_patches')
  Patch_target = os.path.join('/content','mask_patches')
  Patch_rejected = os.path.join('/content','rejected')

  #Here we save the patches, in the /content directory as they will not usually be needed after training
  if os.path.exists(Patch_source):
    shutil.rmtree(Patch_source)
  if os.path.exists(Patch_target):
    shutil.rmtree(Patch_target)
  if os.path.exists(Patch_rejected):
    shutil.rmtree(Patch_rejected)

  os.mkdir(Patch_source)
  os.mkdir(Patch_target)
  os.mkdir(Patch_rejected) #This directory will contain the images that have too little signal.

  patch_num = 0

  for file in tqdm(os.listdir(Training_source)):

    img = io.imread(os.path.join(Training_source, file))
    mask = io.imread(os.path.join(Training_target, file),as_gray=True)

    if DEBUG:
      print(file)
      print(img.dtype)

    # Using view_as_windows with step size equal to the patch size to ensure there is no overlap
    patches_img = view_as_windows(img, (patch_width, patch_height), (patch_width, patch_height))
    patches_mask = view_as_windows(mask, (patch_width, patch_height), (patch_width, patch_height))

    patches_img = patches_img.reshape(patches_img.shape[0]*patches_img.shape[1], patch_width,patch_height)
    patches_mask = patches_mask.reshape(patches_mask.shape[0]*patches_mask.shape[1], patch_width,patch_height)

    if DEBUG:
      print(all_patches_img.shape)
      print(all_patches_img.dtype)

    for i in range(patches_img.shape[0]):
      img_save_path = os.path.join(Patch_source,'patch_'+str(patch_num)+'.tif')
      mask_save_path = os.path.join(Patch_target,'patch_'+str(patch_num)+'.tif')
      patch_num += 1

      # if the mask conatins at least 2% of its total number pixels as mask, then go ahead and save the images
      pixel_threshold_array = sorted(patches_mask[i].flatten())
      if pixel_threshold_array[int(round((len(pixel_threshold_array)-1)*(1-min_fraction)))]>0:
        io.imsave(img_save_path, img_as_ubyte(normalizeMinMax(patches_img[i])))
        io.imsave(mask_save_path, patches_mask[i])
      else:
        io.imsave(Patch_rejected+'/patch_'+str(patch_num)+'_image.tif', img_as_ubyte(normalizeMinMax(patches_img[i])))
        io.imsave(Patch_rejected+'/patch_'+str(patch_num)+'_mask.tif', patches_mask[i])

  return Patch_source, Patch_target


def estimatePatchSize(data_path, max_width = 512, max_height = 512):

  files = os.listdir(data_path)
  
  # Get the size of the first image found in the folder and initialise the variables to that
  n = 0 
  while os.path.isdir(os.path.join(data_path, files[n])):
    n += 1
  (height_min, width_min) = Image.open(os.path.join(data_path, files[n])).size

  # Screen the size of all dataset to find the minimum image size
  for file in files:
    if not os.path.isdir(os.path.join(data_path, file)):
      (height, width) = Image.open(os.path.join(data_path, file)).size
      if width < width_min:
        width_min = width
      if height < height_min:
        height_min = height
  
  # Find the power of patches that will fit within the smallest dataset
  width_min, height_min = (fittingPowerOfTwo(width_min), fittingPowerOfTwo(height_min))

  # Clip values at maximum permissible values
  if width_min > max_width:
    width_min = max_width

  if height_min > max_height:
    height_min = max_height
  
  return (width_min, height_min)

def fittingPowerOfTwo(number):
  n = 0
  while 2**n <= number:
    n += 1 
  return 2**(n-1)

## TODO: create weighted CE for semantic labels
def getClassWeights(Training_target_path):

  Mask_dir_list = os.listdir(Training_target_path)
  number_of_dataset = len(Mask_dir_list)

  class_count = np.zeros(2, dtype=int)
  for i in tqdm(range(number_of_dataset)):
    mask = io.imread(os.path.join(Training_target_path, Mask_dir_list[i]))
    mask = normalizeMinMax(mask)
    class_count[0] += mask.shape[0]*mask.shape[1] - mask.sum()
    class_count[1] += mask.sum()

  n_samples = class_count.sum()
  n_classes = 2

  class_weights = n_samples / (n_classes * class_count)
  return class_weights

def weighted_binary_crossentropy(class_weights):

    def _weighted_binary_crossentropy(y_true, y_pred):
        binary_crossentropy = keras.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * class_weights[1] + (1. - y_true) * class_weights[0]
        weighted_binary_crossentropy = weight_vector * binary_crossentropy

        return keras.mean(weighted_binary_crossentropy)

    return _weighted_binary_crossentropy


def save_augment(datagen,orig_img,dir_augmented_data="/content/augment"):
  """
  Saves a subset of the augmented data for visualisation, by default in /content.

  This is adapted from: https://fairyonice.github.io/Learn-about-ImageDataGenerator.html
  
  """
  try:
    os.mkdir(dir_augmented_data)
  except:
        ## if the preview folder exists, then remove
        ## the contents (pictures) in the folder
    for item in os.listdir(dir_augmented_data):
      os.remove(dir_augmented_data + "/" + item)

    ## convert the original image to array
  x = img_to_array(orig_img)
    ## reshape (Sampke, Nrow, Ncol, 3) 3 = R, G or B
    #print(x.shape)
  x = x.reshape((1,) + x.shape)
    #print(x.shape)
    ## -------------------------- ##
    ## randomly generate pictures
    ## -------------------------- ##
  i = 0
    #We will just save 5 images,
    #but this can be changed, but note the visualisation in 3. currently uses 5.
  Nplot = 5
  for batch in datagen.flow(x,batch_size=1,
                            save_to_dir=dir_augmented_data,
                            save_format='tif',
                            seed=42):
    i += 1
    if i > Nplot - 1:
      break

# Generators
def buildDoubleGenerator(image_datagen, mask_datagen, image_folder_path, mask_folder_path, subset, batch_size, target_size, validatio_split):
  '''
  Can generate image and mask at the same time use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
  
  datagen: ImageDataGenerator 
  subset: can take either 'training' or 'validation'
  '''
  
  # Build the dict for the ImageDataGenerator
  # non_aug_args = dict(width_shift_range = 0,
  #                     height_shift_range = 0,
  #                     rotation_range = 0, #90
  #                     zoom_range = 0,
  #                     shear_range = 0,
  #                     horizontal_flip = False,
  #                     vertical_flip = False,
  #                     fill_mode = 'reflect')
  # default params of data generator is without augmentation
  mask_load_gen = ImageDataGenerator(dtype='uint8', validation_split=validatio_split)
  image_load_gen = ImageDataGenerator(dtype='float32', validation_split=validatio_split, preprocessing_function = normalizePercentile)
  
  image_generator = image_load_gen.flow_from_directory(
        os.path.dirname(image_folder_path),
        classes = [os.path.basename(image_folder_path)],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        subset = subset,
        interpolation = "bicubic",
        seed = 1)
  mask_generator = mask_load_gen.flow_from_directory(
        os.path.dirname(mask_folder_path),
        classes = [os.path.basename(mask_folder_path)],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        subset = subset,
        interpolation = "nearest",
        seed = 1)

  this_generator = zip(image_generator, mask_generator)
  for (img,mask) in this_generator:
      if subset == 'training':
          # Apply the data augmentation
          # the same seed should provide always the same transformation and image loading
          seed = np.random.randint(100000)
          for batch_im in image_datagen.flow(img,batch_size=batch_size, seed=seed):
              break
          mask = mask.astype(np.float32)
          labels = np.unique(mask)
          if len(labels)>1:
              batch_mask = np.zeros_like(mask, dtype='float32')
              for l in range(0, len(labels)):
                  aux = (mask==l).astype(np.float32)
                  for batch_aux in mask_datagen.flow(aux,batch_size=batch_size, seed=seed):
                      break
                  batch_mask += l*(batch_aux>0).astype(np.float32)
              index = np.where(batch_mask>l)
              batch_mask[index]=l
          else:
              batch_mask = mask

          yield (batch_im,batch_mask)

      else:
          yield (img,mask)
      

def prepareGenerators(image_folder_path, mask_folder_path, datagen_parameters, batch_size = 4, target_size = (512, 512), validatio_split = 0.1):
  image_datagen = ImageDataGenerator(**datagen_parameters, preprocessing_function = normalizePercentile)
  mask_datagen = ImageDataGenerator(**datagen_parameters)

  train_datagen = buildDoubleGenerator(image_datagen, mask_datagen, image_folder_path, mask_folder_path, 'training', batch_size, target_size, validatio_split)
  validation_datagen = buildDoubleGenerator(image_datagen, mask_datagen, image_folder_path, mask_folder_path, 'validation', batch_size, target_size, validatio_split)

  return (train_datagen, validation_datagen)


# Normalization functions from Martin Weigert
def normalizePercentile(x, pmin=1, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """This function is adapted from Martin Weigert"""
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):#dtype=np.float32
    """This function is adapted from Martin Weigert"""
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x



# Simple normalization to min/max fir the Mask
def normalizeMinMax(x, dtype=np.float32):
  x = x.astype(dtype,copy=False)
  x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
  return x


# This is code outlines the architecture of U-net. The choice of pooling steps decides the depth of the network. 
def unet(pretrained_weights = None, input_size = (256,256,1), pooling_steps = 4, learning_rate = 1e-4, verbose=True, labels=2):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    # Downsampling steps
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    
    if pooling_steps > 1:
      pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
      conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
      conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)

      if pooling_steps > 2:
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
      
        if pooling_steps > 3:
          pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
          conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
          conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
          drop5 = Dropout(0.5)(conv5)

          #Upsampling steps
          up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
          merge6 = concatenate([drop4,up6], axis = 3)
          conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
          conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
          
    if pooling_steps > 2:
      up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4))
      if pooling_steps > 3:
        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
      merge7 = concatenate([conv3,up7], axis = 3)
      conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
      conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    if pooling_steps > 1:
      up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv3))
      if pooling_steps > 2:
        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
      merge8 = concatenate([conv2,up8], axis = 3)
      conv8 = Conv2D(128, 3, activation= 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
      conv8 = Conv2D(128, 3, activation= 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
      
    if pooling_steps == 1:
      up9 = Conv2D(64, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv2))
    else:
      up9 = Conv2D(64, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8)) #activation = 'relu'
    
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9) #activation = 'relu'
    conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9) #activation = 'relu'
    conv9 = Conv2D(labels, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9) #activation = 'relu'
    conv10 = Conv2D(labels, 1, activation = 'softmax')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = learning_rate), loss = 'sparse_categorical_crossentropy')

    if verbose:
      model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights);

    return model

# Custom callback showing sample prediction
class SampleImageCallback(Callback):

    def __init__(self, model, sample_data, model_path, save=False):
        self.model = model
        self.sample_data = sample_data
        self.model_path = model_path
        self.save = save

    def on_epoch_end(self, epoch, logs={}):
      if np.mod(epoch,5) == 0:
            sample_predict = self.model.predict_on_batch(self.sample_data)

            f=plt.figure(figsize=(16,8))
            plt.subplot(1,labels+1,1)
            plt.imshow(self.sample_data[0,:,:,0], cmap='gray')
            plt.title('Sample source')
            plt.axis('off');
            for i in range(1, labels):
              plt.subplot(1,labels+1,i+1)
              plt.imshow(sample_predict[0,:,:,i], interpolation='nearest', cmap='magma')
              plt.title('Predicted label {}'.format(i))
              plt.axis('off');

            plt.subplot(1,labels+1,labels+1)
            plt.imshow(np.squeeze(np.argmax(sample_predict[0], axis=-1)), interpolation='nearest')
            plt.title('Semantic segmentation')
            plt.axis('off');

            plt.show()

            if self.save:
                plt.savefig(self.model_path + '/epoch_' + str(epoch+1) + '.png')
                random_choice = random.choice(os.listdir(Patch_source))

def predict_as_tiles(Image_path, model):

  # Read the data in and normalize
  Image_raw = io.imread(Image_path, as_gray = True)
  Image_raw = normalizePercentile(Image_raw)

  # Get the patch size from the input layer of the model
  patch_size = model.layers[0].output_shape[0][1:3]
  #patch_size = model.layers[0].output_shape[1:3]

  # Pad the image with zeros if any of its dimensions is smaller than the patch size
  if Image_raw.shape[0] < patch_size[0] or Image_raw.shape[1] < patch_size[1]:
    Image = np.zeros((max(Image_raw.shape[0], patch_size[0]), max(Image_raw.shape[1], patch_size[1])))
    Image[0:Image_raw.shape[0], 0: Image_raw.shape[1]] = Image_raw
  else:
    Image = Image_raw

  # Calculate the number of patches in each dimension
  n_patch_in_width = ceil(Image.shape[0]/patch_size[0])
  n_patch_in_height = ceil(Image.shape[1]/patch_size[1])

  prediction = np.zeros(Image.shape, dtype = 'uint8')

  for x in range(n_patch_in_width):
    for y in range(n_patch_in_height):
      xi = patch_size[0]*x
      yi = patch_size[1]*y

      # If the patch exceeds the edge of the image shift it back 
      if xi+patch_size[0] >= Image.shape[0]:
        xi = Image.shape[0]-patch_size[0]

      if yi+patch_size[1] >= Image.shape[1]:
        yi = Image.shape[1]-patch_size[1]
      
      # Extract and reshape the patch
      patch = Image[xi:xi+patch_size[0], yi:yi+patch_size[1]]
      patch = np.reshape(patch,patch.shape+(1,))
      patch = np.reshape(patch,(1,)+patch.shape)

      # Get the prediction from the patch and paste it in the prediction in the right place
      predicted_patch = model.predict(patch, batch_size = 1)
      prediction[xi:xi+patch_size[0], yi:yi+patch_size[1]] = (np.argmax(np.squeeze(predicted_patch), axis = -1)).astype(np.uint8)


  return prediction[0:Image_raw.shape[0], 0: Image_raw.shape[1]]


def saveResult(save_path, nparray, source_dir_list, prefix=''):
  for (filename, image) in zip(source_dir_list, nparray):
      io.imsave(os.path.join(save_path, prefix+os.path.splitext(filename)[0]+'.tif'), image) # saving as unsigned 8-bit image


def convert2Mask(image, threshold):
  mask = img_as_ubyte(image, force_copy=True)
  mask[mask > threshold] = 255
  mask[mask <= threshold] = 0
  return mask

# -------------- Other definitions -----------
W  = '\033[0m'  # white (normal)
R  = '\033[31m' # red
prediction_prefix = 'Predicted_'


print('-------------------')
print('U-Net and dependencies installed.')

# Colors for the warning messages
class bcolors:
  WARNING = '\033[31m'

# Check if this is the latest version of the notebook

#All_notebook_versions = pd.read_csv("https://raw.githubusercontent.com/HenriquesLab/ZeroCostDL4Mic/master/Colab_notebooks/Latest_Notebook_versions.csv", dtype=str)
#print('Notebook version: '+Notebook_version)
#Latest_Notebook_version = All_notebook_versions[All_notebook_versions["Notebook"] == Network]['Version'].iloc[0]
#print('Latest notebook version: '+Latest_Notebook_version)
#if Notebook_version == Latest_Notebook_version:
#  print("This notebook is up-to-date.")
#else:
#  print(bcolors.WARNING +"A new version of this notebook has been released. We recommend that you download it at https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki")


def pdf_export(trained = False, augmentation = False, pretrained_model = False):
  class MyFPDF(FPDF, HTMLMixin):
    pass

  pdf = MyFPDF()
  pdf.add_page()
  pdf.set_right_margin(-1)
  pdf.set_font("Arial", size = 11, style='B') 

  day = datetime.now()
  datetime_str = str(day)[0:10]

  Header = 'Training report for '+Network+' model ('+model_name+')\nDate: '+datetime_str
  pdf.multi_cell(180, 5, txt = Header, align = 'L') 
    
  # add another cell 
  if trained:
    training_time = "Training time: "+str(hour)+ "hour(s) "+str(mins)+"min(s) "+str(round(sec))+"sec(s)"
    pdf.cell(190, 5, txt = training_time, ln = 1, align='L')
  pdf.ln(1)

  Header_2 = 'Information for your materials and method:'
  pdf.cell(190, 5, txt=Header_2, ln=1, align='L')

  all_packages = ''
  for requirement in freeze(local_only=True):
    all_packages = all_packages+requirement+', '
  #print(all_packages)

  #Main Packages
  main_packages = ''
  version_numbers = []
  for name in ['tensorflow','numpy','Keras']:
    find_name=all_packages.find(name)
    main_packages = main_packages+all_packages[find_name:all_packages.find(',',find_name)]+', '
    #Version numbers only here:
    version_numbers.append(all_packages[find_name+len(name)+2:all_packages.find(',',find_name)])

  cuda_version = subprocess.run('nvcc --version',stdout=subprocess.PIPE, shell=True)
  cuda_version = cuda_version.stdout.decode('utf-8')
  cuda_version = cuda_version[cuda_version.find(', V')+3:-1]
  gpu_name = subprocess.run('nvidia-smi',stdout=subprocess.PIPE, shell=True)
  gpu_name = gpu_name.stdout.decode('utf-8')
  gpu_name = gpu_name[gpu_name.find('Tesla'):gpu_name.find('Tesla')+10]
  #print(cuda_version[cuda_version.find(', V')+3:-1])
  #print(gpu_name)
  loss = str(model.loss)[str(model.loss).find('function')+len('function'):str(model.loss).find('.<')]
  shape = io.imread(Training_source+'/'+os.listdir(Training_source)[1]).shape
  dataset_size = len(os.listdir(Training_source))

  text = 'The '+Network+' model was trained from scratch for '+str(number_of_epochs)+' epochs on '+str(number_of_training_dataset)+' paired image patches (image dimensions: '+str(shape)+', patch size: ('+str(patch_width)+','+str(patch_height)+')) with a batch size of '+str(batch_size)+' and a'+loss+' loss function,'+' using the '+Network+' ZeroCostDL4Mic notebook (v '+Notebook_version[0]+') (von Chamier & Laine et al., 2020). Key python packages used include tensorflow (v '+version_numbers[0]+'), Keras (v '+version_numbers[2]+'), numpy (v '+version_numbers[1]+'), cuda (v '+cuda_version+'). The training was accelerated using a '+gpu_name+'GPU.'

  if pretrained_model:
    text = 'The '+Network+' model was trained for '+str(number_of_epochs)+' epochs on '+str(number_of_training_dataset)+' paired image patches (image dimensions: '+str(shape)+', patch size: ('+str(patch_width)+','+str(patch_height)+')) with a batch size of '+str(batch_size)+'  and a'+loss+' loss function,'+' using the '+Network+' ZeroCostDL4Mic notebook (v '+Notebook_version[0]+') (von Chamier & Laine et al., 2020). The model was re-trained from a pretrained model. Key python packages used include tensorflow (v '+version_numbers[0]+'), Keras (v '+version_numbers[2]+'), numpy (v '+version_numbers[1]+'), cuda (v '+cuda_version+'). The training was accelerated using a '+gpu_name+'GPU.'

  pdf.set_font('')
  pdf.set_font_size(10.)
  pdf.multi_cell(180, 5, txt = text, align='L')
  pdf.set_font('')
  pdf.set_font('Arial', size = 10, style = 'B')
  pdf.ln(1)
  pdf.cell(28, 5, txt='Augmentation: ', ln=1)
  pdf.set_font('')
  if augmentation:
    aug_text = 'The dataset was augmented by'
    if rotation_range != 0:
      aug_text = aug_text+'\n- rotation'
    if horizontal_flip == True or vertical_flip == True:
      aug_text = aug_text+'\n- flipping'
    if zoom_range != 0:
      aug_text = aug_text+'\n- random zoom magnification'
    if horizontal_shift != 0 or vertical_shift != 0:
      aug_text = aug_text+'\n- shifting'
    if shear_range != 0:
      aug_text = aug_text+'\n- image shearing'
  else:
    aug_text = 'No augmentation was used for training.'
  pdf.multi_cell(190, 5, txt=aug_text, align='L')
  pdf.set_font('Arial', size = 11, style = 'B')
  pdf.ln(1)
  pdf.cell(180, 5, txt = 'Parameters', align='L', ln=1)
  pdf.set_font('')
  pdf.set_font_size(10.)
  if Use_Default_Advanced_Parameters:
    pdf.cell(200, 5, txt='Default Advanced Parameters were enabled')
  pdf.cell(200, 5, txt='The following parameters were used for training:')
  pdf.ln(1)
  html = """ 
  <table width=40% style="margin-left:0px;">
    <tr>
      <th width = 50% align="left">Parameter</th>
      <th width = 50% align="left">Value</th>
    </tr>
    <tr>
      <td width = 50%>number_of_epochs</td>
      <td width = 50%>{0}</td>
    </tr>
    <tr>
      <td width = 50%>patch_size</td>
      <td width = 50%>{1}</td>
    </tr>
    <tr>
      <td width = 50%>batch_size</td>
      <td width = 50%>{2}</td>
    </tr>
    <tr>
      <td width = 50%>number_of_steps</td>
      <td width = 50%>{3}</td>
    </tr>
    <tr>
      <td width = 50%>percentage_validation</td>
      <td width = 50%>{4}</td>
    </tr>
    <tr>
      <td width = 50%>initial_learning_rate</td>
      <td width = 50%>{5}</td>
    </tr>
    <tr>
      <td width = 50%>pooling_steps</td>
      <td width = 50%>{6}</td>
    </tr>
    <tr>
      <td width = 50%>min_fraction</td>
      <td width = 50%>{7}</td>
  </table>
  """.format(number_of_epochs, str(patch_width)+'x'+str(patch_height), batch_size, number_of_steps, percentage_validation, initial_learning_rate, pooling_steps, min_fraction)
  pdf.write_html(html)

  #pdf.multi_cell(190, 5, txt = text_2, align='L')
  pdf.set_font("Arial", size = 11, style='B')
  pdf.ln(1)
  pdf.cell(190, 5, txt = 'Training Dataset', align='L', ln=1)
  pdf.set_font('')
  pdf.set_font('Arial', size = 10, style = 'B')
  pdf.cell(29, 5, txt= 'Training_source:', align = 'L', ln=0)
  pdf.set_font('')
  pdf.multi_cell(170, 5, txt = Training_source, align = 'L')
  pdf.set_font('')
  pdf.set_font('Arial', size = 10, style = 'B')
  pdf.cell(28, 5, txt= 'Training_target:', align = 'L', ln=0)
  pdf.set_font('')
  pdf.multi_cell(170, 5, txt = Training_target, align = 'L')
  #pdf.cell(190, 5, txt=aug_text, align='L', ln=1)
  pdf.ln(1)
  pdf.set_font('')
  pdf.set_font('Arial', size = 10, style = 'B')
  pdf.cell(21, 5, txt= 'Model Path:', align = 'L', ln=0)
  pdf.set_font('')
  pdf.multi_cell(170, 5, txt = model_path+'/'+model_name, align = 'L')
  pdf.ln(1)
  pdf.cell(60, 5, txt = 'Example Training pair', ln=1)
  pdf.ln(1)
  exp_size = io.imread('/content/TrainingDataExample_Unet2D.png').shape
  pdf.image('/content/TrainingDataExample_Unet2D.png', x = 11, y = None, w = round(exp_size[1]/8), h = round(exp_size[0]/8))
  pdf.ln(1)
  ref_1 = 'References:\n - ZeroCostDL4Mic: von Chamier, Lucas & Laine, Romain, et al. "Democratising deep learning for microscopy with ZeroCostDL4Mic." Nature Communications (2021).'
  pdf.multi_cell(190, 5, txt = ref_1, align='L')
  ref_2 = '- Unet: Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.'
  pdf.multi_cell(190, 5, txt = ref_2, align='L')
  # if Use_Data_augmentation:
  #   ref_3 = '- Augmentor: Bloice, Marcus D., Christof Stocker, and Andreas Holzinger. "Augmentor: an image augmentation library for machine learning." arXiv preprint arXiv:1708.04680 (2017).'
  #   pdf.multi_cell(190, 5, txt = ref_3, align='L')
  pdf.ln(3)
  reminder = 'Important:\nRemember to perform the quality control step on all newly trained models\nPlease consider depositing your training dataset on Zenodo'
  pdf.set_font('Arial', size = 11, style='B')
  pdf.multi_cell(190, 5, txt=reminder, align='C')

  pdf.output(model_path+'/'+model_name+'/'+model_name+'_training_report.pdf')

  print('------------------------------')
  print('PDF report exported in '+model_path+'/'+model_name+'/')

def qc_pdf_export():
  class MyFPDF(FPDF, HTMLMixin):
    pass

  pdf = MyFPDF()
  pdf.add_page()
  pdf.set_right_margin(-1)
  pdf.set_font("Arial", size = 11, style='B') 

  Network = 'Unet 2D'

  day = datetime.now()
  datetime_str = str(day)[0:10]

  Header = 'Quality Control report for '+Network+' model ('+QC_model_name+')\nDate: '+datetime_str
  pdf.multi_cell(180, 5, txt = Header, align = 'L') 

  all_packages = ''
  for requirement in freeze(local_only=True):
    all_packages = all_packages+requirement+', '

  pdf.set_font('')
  pdf.set_font('Arial', size = 11, style = 'B')
  pdf.ln(2)
  pdf.cell(190, 5, txt = 'Loss curves', ln=1, align='L')
  pdf.ln(1)
  exp_size = io.imread(full_QC_model_path+'/Quality Control/QC_example_data.png').shape
  if os.path.exists(full_QC_model_path+'/Quality Control/lossCurvePlots.png'):
    pdf.image(full_QC_model_path+'/Quality Control/lossCurvePlots.png', x = 11, y = None, w = round(exp_size[1]/12), h = round(exp_size[0]/3))
  else:
    pdf.set_font('')
    pdf.set_font('Arial', size=10)
    pdf.multi_cell(190, 5, txt='If you would like to see the evolution of the loss function during training please play the first cell of the QC section in the notebook.',align='L')
  pdf.ln(2)
  pdf.set_font('')
  pdf.set_font('Arial', size = 10, style = 'B')
  pdf.ln(3)
  pdf.cell(80, 5, txt = 'Example Quality Control Visualisation', ln=1)
  pdf.ln(1)
  exp_size = io.imread(full_QC_model_path+'/Quality Control/QC_example_data.png').shape
  pdf.image(full_QC_model_path+'/Quality Control/QC_example_data.png', x = 16, y = None, w = round(exp_size[1]/8), h = round(exp_size[0]/8))
  pdf.ln(1)
  pdf.set_font('')
  pdf.set_font('Arial', size = 11, style = 'B')
  pdf.ln(1)
  pdf.cell(180, 5, txt = 'Quality Control Metrics', align='L', ln=1)
  pdf.set_font('')
  pdf.set_font_size(10.)

  pdf.ln(1)
  html = """
  <body>
  <font size="10" face="Courier New" >
  <table width=60% style="margin-left:0px;">"""
  with open(full_QC_model_path+'/Quality Control/QC_metrics_'+QC_model_name+'.csv', 'r') as csvfile:
    metrics = csv.reader(csvfile)
    header = next(metrics)
    image = header[0]
    IoU = header[-1]
    header = """
    <tr>
    <th width = 33% align="center">{0}</th>
    <th width = 33% align="center">{1}</th>
    </tr>""".format(image,IoU)
    html = html+header
    i=0
    for row in metrics:
      i+=1
      image = row[0]
      IoU = row[-1]
      cells = """
        <tr>
          <td width = 33% align="center">{0}</td>
          <td width = 33% align="center">{1}</td>
        </tr>""".format(image,str(round(float(IoU),3)))
      html = html+cells
    html = html+"""</body></table>"""
    
  pdf.write_html(html)

  pdf.ln(1)
  pdf.set_font('')
  pdf.set_font_size(10.)
  ref_1 = 'References:\n - ZeroCostDL4Mic: von Chamier, Lucas & Laine, Romain, et al. "Democratising deep learning for microscopy with ZeroCostDL4Mic." Nature Communications (2021).'
  pdf.multi_cell(190, 5, txt = ref_1, align='L')
  ref_2 = '- Unet: Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.'
  pdf.multi_cell(190, 5, txt = ref_2, align='L')

  pdf.ln(3)
  reminder = 'To find the parameters and other information about how this model was trained, go to the training_report.pdf of this model which should be in the folder of the same name.'

  pdf.set_font('Arial', size = 11, style='B')
  pdf.multi_cell(190, 5, txt=reminder, align='C')

  pdf.output(full_QC_model_path+'/Quality Control/'+QC_model_name+'_QC_report.pdf')

  print('------------------------------')
  print('QC PDF report exported as '+full_QC_model_path+'/Quality Control/'+QC_model_name+'_QC_report.pdf')

# Build requirements file for local run
after = [str(m) for m in sys.modules]
build_requirements_file(before, after)

# from tensorflow.python.client import device_lib 
# device_lib.list_local_devices()

# print the tensorflow version
print('Tensorflow version is ' + str(tf.__version__))


# ### Load filoTips requirements

# In[3]:


subprocess.run('pip install --upgrade --no-cache-dir gdown',shell=True)
install('opencv-python')
import pandas as pd
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import tifffile as tiff
import os
import seaborn as sns
#import statistics as stats
import math
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
install('researchpy')
import researchpy as rp
import shutil


# ### Download the deep learning agent and prepare for predictions

# In[4]:


os.chdir(path)
if not os.path.exists('filoTips analysis'):
    os.makedirs('filoTips analysis')
os.chdir(path+'/filoTips analysis')
if not os.path.exists('filoTips predictions'):
  os.makedirs('filoTips predictions')
if not os.path.exists('filoTips source'):
  os.makedirs('filoTips source')

os.chdir(path)
files=sorted(glob.glob('*.tif'))
if len(files)>0:
  for file in files:
    temp=file.replace('.tif','.tiff')
    os.replace(file,temp)
files=sorted(glob.glob('*.tiff'))
for file in files:
  shutil.copy(file,path+'/filoTips analysis/filoTips source/'+file)

os.chdir(path+'/filoTips analysis')
if model_type=='Default':
  subprocess.run('gdown --id 1zh5j_VL380ebyQxDCJ5N2fhOxWqkFQsC',shell=True)
  subprocess.run('tar -xf StanleyV3.zip',shell=True)
if model_type=='Custom':
  subprocess.run(['gdown',custom_model_DriveLink])
  zipp=glob.glob('*.zip')[0]
  custom_model_name = zipp.replace('.zip','')
  subprocess.run(['tar', '-xf', zipp])

os.chdir('filoTips source')
for file in files:
  img=cv2.imread(file,-1)
  img=cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  tiff.imsave(file,img)
  
os.chdir(path+'/filoTips analysis')


# ### Have the model generate masks of cell bodies, filopodia, and background (includes code from "U-Net_2D_Multilabel" ZeroCostDL4Mic notebook)

# In[5]:


Data_folder = path+'/filoTips analysis/filoTips source'
Results_folder = path+'/filoTips analysis/filoTips predictions'
if model_type=='Default':
  Prediction_model_folder = path+"/filoTips analysis/Stanley"
if model_type=='Custom':
  Prediction_model_folder = path+'/filoTips analysis/'+custom_model_name

Use_the_current_trained_model = False

#Here we find the loaded model name and parent path
Prediction_model_name = os.path.basename(Prediction_model_folder)
Prediction_model_path = os.path.dirname(Prediction_model_folder)


# ------------- Failsafes ------------
if (Use_the_current_trained_model): 
  print("Using current trained network")
  Prediction_model_name = model_name
  Prediction_model_path = model_path

full_Prediction_model_path = os.path.join(Prediction_model_path, Prediction_model_name)
if os.path.exists(full_Prediction_model_path):
  print("The "+Prediction_model_name+" network will be used.")
else:
  print(R+'!! WARNING: The chosen model does not exist !!'+W)
  print('Please make sure you provide a valid model path and model name before proceeding further.')


# ------------- Prepare the model and run predictions ------------

# Load the model and prepare generator



unet = load_model(os.path.join(Prediction_model_path, Prediction_model_name, 'weights_best.hdf5'), custom_objects={'_weighted_binary_crossentropy': weighted_binary_crossentropy(np.ones(2))})
#Input_size = unet.layers[0].output_shape[1:3]
Input_size= unet.layers[0].output_shape[0][1:3]
print('Model input size: '+str(Input_size[0])+'x'+str(Input_size[1]))


# Create a list of sources
source_dir_list = os.listdir(Data_folder)
number_of_dataset = len(source_dir_list)
print('Number of dataset found in the folder: '+str(number_of_dataset))

predictions = []
for i in range(number_of_dataset):
  predictions.append(predict_as_tiles(os.path.join(Data_folder, source_dir_list[i]), unet))
  #predictions.append(prediction(os.path.join(Data_folder, source_dir_list[i]), os.path.join(Prediction_model_path, Prediction_model_name)))


# Save the results in the folder along with the masks according to the set threshold
saveResult(Results_folder, predictions, source_dir_list, prefix=prediction_prefix)


# ------------- For display ------------
print('--------------------------------------------------------------')
os.chdir(path+'/filoTips analysis/filoTips predictions')
files=sorted(glob.glob('*.tif'))
for file in files:
  name=file.replace('.tif','.tiff')
  os.rename(file,name)
os.chdir(path+'/filoTips analysis')

def show_prediction_mask(file=os.listdir(Data_folder)):

  plt.figure(figsize=(10,6))
  # Wide-field
  plt.subplot(1,2,1)
  plt.axis('off')
  img_Source = plt.imread(os.path.join(Data_folder, file))
  plt.imshow(img_Source, cmap='gray')
  plt.title('Source image',fontsize=15)
  # Prediction
  plt.subplot(1,2,2)
  plt.axis('off')
  img_Prediction = plt.imread(os.path.join(Results_folder, prediction_prefix+file))
  plt.imshow(img_Prediction, cmap='gray')
  plt.title('Prediction',fontsize=15)

#interact(show_prediction_mask);



# ### Use the predictions to get information about cell bodies and filopodia

# In[6]:


#@title 5) Use the masks to get information about cell bodies and filopodia
print('')
print('-- Analyzing images--')
def get_distance(x2,x1,y2,y1):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

os.chdir(path+'/filoTips analysis/filoTips predictions')
temp=sorted(glob.glob('*.tiff'))
masks,images=[],[]
for mask in temp:
  masks.append(mask.replace('Predicted_',''))
  images.append(mask.replace('Predicted_',''))
  new_mask=(mask.replace('Predicted_',''))
  os.rename(mask,new_mask)

for q in range(0,len(images)):
  print('--Analyzing '+str(q+1)+' / '+str(len(images))+' images--')
  os.chdir(path+'/filoTips analysis/filoTips predictions')
  mask=cv2.imread(images[q],-1)
  os.chdir(path+'/filoTips analysis/filoTips source')
  img=cv2.imread(images[q],-1)
  img_copy=cv2.imread(images[q])
  img_copy[np.where(img_copy>0)]=0
  img_copy2=np.copy(img_copy)
  img_copy3=np.copy(img_copy)
  img_copy4=np.copy(img_copy)
  cell=np.copy(mask)
  cell[np.where(cell==2)]=0
  filo_tip=np.copy(mask)
  filo_tip[np.where(filo_tip==1)]=0

  contours, hierarchy = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  mean_body_vals,mean_cortex_vals,centroids,areas,perimeters,aspect_ratios,leading_edge_vals,leading_body,cortex_body,all_body_x,all_body_y,all_cortex_x,all_cortex_y,cortex_means,body_means,lead_means,centroids,side_rear_means,body_sums=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
  white,yellow,blue,green,pink,red,orange,black,light_orange,gray=(255,255,255),(204,204,0),(51,153,255),(0,204,0),(255,0,127),(255,51,51),(153,76,0),(0,0,0),(255,178,102),(160,160,160)
  gray1,gray2,gray3,gray4,purple=(224,224,224),(192,192,192),(160,160,160),(128,128,128),(102,0,51)
  for i in range(0,len(contours)):
      M=cv2.moments(contours[i])
      if M['m00']!=0:
          area=cv2.contourArea(contours[i])/(pixel_micron**2)
          if area>25:
              centroid=(int(M['m10']/M['m00']),int(M['m01']/M['m00']))
              centroids.append(centroid)
              areas.append(cv2.contourArea(contours[i])/(pixel_micron**2))
              perimeters.append(cv2.arcLength(contours[i],True)/(pixel_micron))
              rect=cv2.minAreaRect(contours[i])
              wh=rect[1]
              w=np.min(wh)
              h=np.max(wh)
              aspect_ratios.append(float(w)/h)
              img_copy2=cv2.drawContours(np.copy(img_copy),contours,i,white,-1)
              img_copy2=cv2.drawContours(img_copy2,contours,i,black,20)#was15
              img_copy2=np.where((img_copy2==list(white)).all(axis=2))
              img_copy8=cv2.drawContours(np.copy(img_copy),contours,i,purple,-1)
              img_copy8=cv2.drawContours(img_copy8,contours,i,black,9)
              img_copy8=np.where((img_copy8==list(purple)).all(axis=2))
              img_copy3=cv2.drawContours(np.copy(img_copy3),contours,i,gray1,-1)
              img_copy3=cv2.drawContours(img_copy3,contours,i,black,5)
              img_copy3[img_copy8]=purple
              img_copy3[img_copy2]=gray2

              body_vals=img[np.where((img_copy3==list(gray2)).all(axis=2))]
              body_means.append(np.mean(body_vals))
              body_sums.append(np.sum(body_vals))
              cortex_vals=img[np.where((img_copy3==list(gray1)).all(axis=2))]
              cortex_means.append(np.mean(cortex_vals))
              lead_ind=int(np.argmax(cortex_vals==max(cortex_vals)))
              cortex_coords=np.where((img_copy3==list(gray1)).all(axis=2))
              y1,x1=np.where((img_copy3==list(gray1)).all(axis=2))[0],np.where((img_copy3==list(gray1)).all(axis=2))[1]
              img_copy4=cv2.circle(img_copy4, (x1[lead_ind],y1[lead_ind]), 15, white,thickness=-1)
              circle_coords=np.where((img_copy4==list(white)).all(axis=2))
              circle=[]
              leading_coords=[]
              lead_x,lead_y=[],[]
              for y in range(0,len(circle_coords[0])):
                  circle.append((circle_coords[0][y],circle_coords[1][y]))
              for x in range(0,len(cortex_coords[0])):
                  c_coords=(cortex_coords[0][x],cortex_coords[1][x])
                  if c_coords in circle:
                      lead_x.append(c_coords[1])
                      lead_y.append(c_coords[0])
              lead_coord=lead_y,lead_x
              img_copy3[lead_coord]=gray3        
              lead_vals=img[lead_coord]        
              lead_means.append(np.mean(lead_vals))
              img_copy4=cv2.circle(img_copy4, (x1[lead_ind],y1[lead_ind]), 15, gray1,thickness=-1)
              side_rear_means.append(np.mean(img[np.where((img_copy3==list(gray1)).all(axis=2))]))
              img_copy3[np.where((img_copy3==list(gray1)).all(axis=2))]=orange
              img_copy3[np.where((img_copy3==list(gray2)).all(axis=2))]=yellow
              img_copy3[np.where((img_copy3==list(gray3)).all(axis=2))]=blue
              img_copy3[np.where((img_copy3==list(purple)).all(axis=2))]=gray4
  img_copy5=np.copy(img_copy3)            
  img_copy5[np.where((img_copy3==list(orange)).all(axis=2))]=blue
  filo_tip[np.where((img_copy3==list(yellow)).all(axis=2))]=0


  #get filo information
  contours, hierarchy = cv2.findContours(filo_tip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  filo_centroids,filo_lengths_pix,filo_lengths_um,filo_means,temp_filo,cell_assignment,filo_areas,filo_sums=[],[],[],[],[],[],[],[]
  for i in range(0,len(contours)):
      M=cv2.moments(contours[i])
      if M['m00']!=0:
          centroid=(int(M['m10']/M['m00']),int(M['m01']/M['m00']))
          all_blue=np.where((img_copy5==list(blue)).all(axis=2))
          distances=[]
          for d in range(0,len(all_blue[0])):
              y2,y1,x2,x1=centroid[1],all_blue[0][d],centroid[0],all_blue[1][d]
              distance=get_distance(x2, x1, y2, y1)
              distances.append(distance)
          if len(distances)==0:
            break
          ind=distances.index(min(distances))
          length=min(distances)/pixel_micron

          if length <10:
              filo_centroids.append(centroid)
              filo_lengths_pix.append(min(distances))
              filo_lengths_um.append(min(distances)/pixel_micron)
              area=cv2.contourArea(contours[i])/(pixel_micron**2)
              img_copy3=cv2.circle(img_copy3,centroid,radius=1,color=white,thickness=-1,)

              y=np.where((img_copy3==list(white)).all(axis=2))[0]
              x=np.where((img_copy3==list(white)).all(axis=2))[1]
              filo_means.append(np.mean(img[y,x]))
              filo_sums.append(np.sum(img[y,x]))
              filo_areas.append(area)
              distances=[]
              for cent in centroids:
                  y2,y1,x2,x1=centroid[1],cent[1],centroid[0],cent[0]
                  distances.append(get_distance(x2, x1, y2, y1))   
              cell_assignment.append(distances.index(min(distances))+1)
              cell_assignments=distances.index(min(distances))  
              img_copy3[y,x]=pink
              temp_filo.append(img[y,x])
              coord=(all_blue[1][ind],all_blue[0][ind])
              img_copy5=cv2.circle(img_copy5,coord,1,red)
              img_copy5=cv2.line(img_copy5,coord,centroids[cell_assignments],red,2)
              img_copy3=cv2.line(img_copy3,centroid,coord,pink,1)

  #calculate spacing (filoSpace)
  if filoSpace == True:
    img_copy6=np.copy(img_copy5)
    img_copy7=np.copy(img_copy)
    img_copy6[np.where((img_copy5!=list(blue)).all(axis=2))]=black
    img_copy6 = cv2.cvtColor(img_copy6, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(img_copy6, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cell_ind=[]
    space_perimeters=[]
    for i in range(0,len(contours)):
      M=cv2.moments(contours[i])
      if M['m00']!=0:
        perimeter=cv2.arcLength(contours[i],False)/(pixel_micron)
        centroid=(int(M['m10']/M['m00']),int(M['m01']/M['m00']))
        if perimeter<20:
          img_copy7=cv2.drawContours(np.copy(img_copy7),contours,i,white,-1)
          distances2=[]
          for d in range(0,len(centroids)):
            x1,y1,x2,y2=centroid[0],centroid[1],centroids[d][0],centroids[d][1]
            distance=get_distance(x2, x1, y2, y1)
            distances2.append(distance)
          cell_ind.append(distances2.index(min(distances2))+1)
          space_perimeters.append(perimeter)
    cell_ind=np.asarray(cell_ind)
    space_perimeters=np.asarray(space_perimeters)
    inds=np.asarray(list(set(cell_ind)))
    space_cell_num,space_cell_avg=[],[]
    for i in inds:
      cell_inds=np.where(cell_ind==i)[0]
      space_cell_num.append(i)
      space_cell_avg.append(np.mean(space_perimeters[cell_inds]))
  
  #plot centroids and cell numbers
  k=0
  for centroid in centroids:
      k+=1
      img_copy3=cv2.circle(img_copy3,centroid,5,green,2)
      img_copy3=cv2.putText(img_copy3,text=str(k),org=centroid,fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(254,254,254), thickness=2, lineType=cv2.LINE_AA)

  #calculate cortex/cell and filo/cell ratios
  cortex_body= [i/j for i,j in zip(cortex_means,body_means)]
  lead_body=[i/j for i,j in zip(lead_means,body_means)]
  side_rear_body=[i/j for i,j in zip(side_rear_means,body_means)]
  lead_side_rear=[i/j for i,j in zip(lead_means,side_rear_means)]
  final_assignments=[]
  for i in range(1,len(centroids)+1):
      if i not in cell_assignment:
          final_assignments.append(0)
      if i in cell_assignment:
          final_assignments.append(cell_assignment.count(i))
  cell_num,filo_num=list(range(1,len(centroids)+1)),list(range(1,len(filo_centroids)+1))
  filo_cell_vals=[]
  for cell in cell_assignment:
      filo_cell_vals.append(body_means[cell-1])
  filo_body=[i/j for i,j in zip(filo_means,filo_cell_vals)]
  filoDensity = [i/j*10 for i,j in zip(final_assignments,perimeters)]

  #include spacing summary information
  if filoSpace==True:
    for i in cell_num:
      if i not in space_cell_num:
        space_cell_num.append(i)
        space_cell_avg.append('nan')
    zipped_lists = zip(space_cell_num, space_cell_avg)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    space_cell_num, space_cell_avg = [ list(tuple) for tuple in  tuples]

  #make output file and pool analysis
  os.chdir(path+'/filoTips analysis')
  if not os.path.exists('filoTips_Output'):
      os.makedirs('filoTips_Output')
  exp_num_cell=np.repeat(images[q],len(cell_num))
  exp_num_filo=np.repeat(images[q],len(filo_num))
  if filoSpace==True:
    cell_dict={'Experiment Name':exp_num_cell,'Cell Number':cell_num,'Aspect Ratio':aspect_ratios,'Body Intensity Sum':body_sums,'Body Intensity':body_means,'Cortex Intensity':cortex_means,'Leading Edge Intensity':lead_means,'Side & Rear Intensity':side_rear_means,'Cortex_Body':cortex_body,'Leading Edge_Body':lead_body,'SideRear_Body':side_rear_body,'Leading Edge_SideRear':lead_side_rear,'Filo Number':final_assignments,'Filos/10um':filoDensity,'Cell Area (um^2)':areas,'Perimeter (um)':perimeters,'Average Inter-filo Distance (um)':space_cell_avg}
  else:
    cell_dict={'Experiment Name':exp_num_cell,'Cell Number':cell_num,'Aspect Ratio':aspect_ratios,'Body Intensity Sum':body_sums,'Body Intensity':body_means,'Cortex Intensity':cortex_means,'Leading Edge Intensity':lead_means,'Side & Rear Intensity':side_rear_means,'Cortex_Body':cortex_body,'Leading Edge_Body':lead_body,'SideRear_Body':side_rear_body,'Leading Edge_SideRear':lead_side_rear,'Filo Number':final_assignments,'Filos/10um':filoDensity,'Cell Area (um^2)':areas,'Perimeter (um)':perimeters}
  filo_dict={'Experiment Name':exp_num_filo,'Filo Number':filo_num,'Cell Assignment':cell_assignment,'Tip Intensity Sum':filo_sums,'Filo Tip Intensity':filo_means,'Assigned Cell Body Intensity':filo_cell_vals,'Filo Tip Area (um^2)':filo_areas,'Filo/Body':filo_body,'Filo Length(um)':filo_lengths_um}
  cell_output=pd.DataFrame(cell_dict)
  filo_output=pd.DataFrame(filo_dict)
  cell_output.to_csv('filoTips_Output/'+str(images[q])+'_Cell_Output_.csv',index=False)
  filo_output.to_csv('filoTips_Output/'+str(images[q])+'_Filo_Output_.csv',index=False)

  if filoSpace==True:    
    fig, final=plt.subplots(1,3)
    final[0].imshow(img)
    final[0].set_title('Original Image')
    final[0].axis('off')
    final[1].imshow(img_copy3)
    final[1].set_title('filoTips')
    final[1].axis('off')
    final[2].imshow(img_copy5)
    final[2].set_title('filoSpace')
    final[2].axis('off')
    plt.savefig('filoTips_Output/'+str(images[q])+'_Annotation.tiff',dpi=Annotation_DPI)
  else:
    fig, final=plt.subplots(1,2)
    final[0].imshow(img)
    final[0].set_title('Original Image')
    final[0].axis('off')
    final[1].imshow(img_copy3)
    final[1].set_title('filoTips')
    final[1].axis('off')
    plt.savefig('filoTips_Output/'+str(images[q])+'_Annotation.tiff',dpi=Annotation_DPI)


# ### Collect and organize the analysis

# In[7]:


#@title 6) Pool together the analysis
os.chdir(path+'/filoTips analysis')
cell_outputs=sorted(glob.glob("filoTips_Output/*Cell_Output_.csv"))
filo_outputs=sorted(glob.glob('filoTips_Output/*Filo_Output_.csv'))
cell_dfs = (pd.read_csv(f) for f in cell_outputs)
cell_dfs = pd.concat(cell_dfs, ignore_index=True)
filo_dfs = (pd.read_csv(f) for f in filo_outputs)
filo_dfs = pd.concat(filo_dfs, ignore_index=True)
cell_dfs.to_csv('filoTips_Output/Total_Cell_Output.csv',index=False)
filo_dfs.to_csv('filoTips_Output/Total_Filo_Output.csv',index=False)


# ### Data visualization (if statistical summary enabled)

# In[8]:


#@title 7) Optional data visualization
cell_file=pd.read_csv(path+'/filoTips analysis/filoTips_Output/Total_Cell_Output.csv')

if not os.path.exists(path+'/filoTips analysis/filoTips_Output/Plots'):
        os.makedirs(path+'/filoTips analysis/filoTips_Output/Plots')

if comparative_analysis ==True:
  prompt1=[Condition_1,Condition_2,Condition_3,Condition_4]
  if '' in prompt1:
    prompt1.remove('')
    if '' in prompt1:
      prompt1.remove('')
  prompt1=str(len(prompt1))
  parameter=['Aspect Ratio','Cortex_Body','Leading Edge_Body','SideRear_Body','Leading Edge_Body','Filo Number','Cell Area (um^2)','Perimeter (um)']

  data= cell_file
  for t in range(0,len(parameter)):
    if prompt1=='2':
      #setup
      #data=data.dropna()
      if 'Condition' not in data.columns:
          ind1=data.loc[data['Experiment Name'].str.contains(Condition_1)].index
          ind2=data.loc[data['Experiment Name'].str.contains(Condition_2)].index
          data.insert(0,'Condition',0)
          data['Condition'][ind1]=Condition_1
          data['Condition'][ind2]=Condition_2
      set1=data.loc[data['Condition'] == Condition_1].reset_index(drop=True)
      set2=data.loc[data['Condition'] == Condition_2].reset_index(drop=True)
      
      print('')
      print('--'+parameter[t]+'--')
      print('-----Mann-Whitney U Test-----')
      kruskal=stats.mannwhitneyu(set1[parameter[t]],set2[parameter[t]])
      print(kruskal)
      print(' ')

      #t-test
      print('--'+parameter[t]+'--')
      print('-----T-test-----')
      t_test=rp.ttest(group1=set1[parameter[t]],group1_name=Condition_1,
              group2=set2[parameter[t]],group2_name=Condition_2)
      print(t_test)
      means=np.round([t_test[0]['Mean'][0],t_test[0]['Mean'][1]],decimals=2)
      stds=np.round([t_test[0]['SD'][0],t_test[0]['SD'][1]],decimals=2)
      bars=t_test[0]['Variable'][0],t_test[0]['Variable'][1]
      x_pos=x_pos=list(np.arange(len(bars)))
      means_stds=[]
      for i in range(0,len(means)):
          means_stds.append(str(means[i])+'±'+str(stds[i]))
      nums=[len(set1[parameter[t]]),len(set2[parameter[t]])]
      
      #violin plot
      pal=sns.color_palette()
      palp=sns.color_palette("husl",8)
      cols=[palp[3],palp[5]]
      dpi=150
      fig,ax1=plt.subplots()
      sns.violinplot(data = data[['Condition',parameter[t]]], x=parameter[t], y="Condition", order=[Condition_1,Condition_2], palette=cols, showmeans=True,inner=None)
      ax1.set_xlabel(parameter[t])
      for i in range(len(means_stds)):
          ax1.annotate(str(means_stds[i]+'\nn='+str(nums[i])),xy=(means[i],i),horizontalalignment='center',verticalalignment='center')
      ax1.xaxis.set_label_position('top')
      ax1.xaxis.tick_top()
      plt.show()
      temp=parameter[t].split('(')[0]
      fig.savefig('filoTips_Output/Plots/'+temp+ '.tiff',dpi=dpi,bbox_inches='tight')
      
    if prompt1=='3':
      #setup
      #data=data.dropna()
      if 'Condition' not in data.columns:
          ind1=data.loc[data['Experiment Name'].str.contains(Condition_1)].index
          ind2=data.loc[data['Experiment Name'].str.contains(Condition_2)].index
          ind3=data.loc[data['Experiment Name'].str.contains(Condition_3)].index
          data.insert(0,'Condition',0)
          data['Condition'][ind1]=Condition_1
          data['Condition'][ind2]=Condition_2
          data['Condition'][ind3]=Condition_3
      set1=data.loc[data['Condition'] == Condition_1].reset_index(drop=True)
      set2=data.loc[data['Condition'] == Condition_2].reset_index(drop=True)
      set3=data.loc[data['Condition'] == Condition_3].reset_index(drop=True)
      
      #kruskal-wallis
      print('-----Kruskal-Wallis Test-----')
      kruskal=stats.kruskal(set1[parameter[t]],set2[parameter[t]],set3[parameter[t]])
      print(kruskal)
      print(' ')
      
      #ANOVA
      print('-----ANOVA-----')
      an=stats.f_oneway(set1[parameter[t]],set2[parameter[t]],set3[parameter[t]])
      vals=data[parameter[t]].tolist()
      names=data['Condition'].tolist()
      tukey=pairwise_tukeyhsd(endog=vals,
                              groups=names,
                              alpha=0.05)
      print(tukey)
      means=np.round([np.mean(set1[parameter[t]]),np.mean(set2[parameter[t]]),np.mean(set3[parameter[t]])],decimals=2)
      stds=np.round([np.std(set1[parameter[t]]),np.std(set2[parameter[t]]),np.std(set3[parameter[t]])],decimals=2)
      means_stds=[]
      for i in range(0,len(means)):
          means_stds.append(str(means[i])+'±'+str(stds[i]))
      nums=[len(set1[parameter[t]]),len(set2[parameter[t]]),len(set3[parameter[t]])]
      
      #violin plot
      pal=sns.color_palette()
      palp=sns.color_palette("husl",8)
      cols=[palp[3],palp[5],palp[1]]
      dpi=150
      fig,ax1=plt.subplots()
      sns.violinplot(data = data[['Condition',parameter[t]]], x=parameter[t], y="Condition", order=[Condition_1,Condition_2,Condition_3], palette=cols, showmeans=True,inner=None)
      ax1.set_xlabel(parameter[t])
      for i in range(len(means_stds)):
          ax1.annotate(str(means_stds[i]+'\nn='+str(nums[i])),xy=(means[i],i),horizontalalignment='center',verticalalignment='center')
      ax1.xaxis.set_label_position('top')
      ax1.xaxis.tick_top()
      plt.show()
      temp=parameter[t].split('(')[0]
      fig.savefig('filoTips_Output/Plots/'+temp+ '.tiff',dpi=dpi,bbox_inches='tight')
      
      
    if prompt1=='4':
      #setup
      #data=data.dropna()
      if 'Condition' not in data.columns:
          ind1=data.loc[data['Experiment Name'].str.contains(Condition_1)].index
          ind2=data.loc[data['Experiment Name'].str.contains(Condition_2)].index
          ind3=data.loc[data['Experiment Name'].str.contains(Condition_3)].index
          ind4=data.loc[data['Experiment Name'].str.contains(Condition_4)].index
          data.insert(0,'Condition',0)
          data['Condition'][ind1]=Condition_1
          data['Condition'][ind2]=Condition_2
          data['Condition'][ind3]=Condition_3
          data['Condition'][ind4]=Condition_4
      set1=data.loc[data['Condition'] == Condition_1].reset_index(drop=True)
      set2=data.loc[data['Condition'] == Condition_2].reset_index(drop=True)
      set3=data.loc[data['Condition'] == Condition_3].reset_index(drop=True)
      set4=data.loc[data['Condition'] == Condition_4].reset_index(drop=True)
    
      #kruskal-wallis
      print('-----Kruskal-Wallis Test-----')
      kruskal=stats.kruskal(set1[parameter[t]],set2[parameter[t]],set3[parameter[t]],set4[parameter[t]])
      print(kruskal)
      print(' ')
      
      #ANOVA
      print('-----ANOVA-----')
      an=stats.f_oneway(set1[parameter[t]],set2[parameter[t]],set3[parameter[t]],set4[parameter[t]])
      vals=data[parameter[t]].tolist()
      names=data['Condition'].tolist()
      tukey=pairwise_tukeyhsd(endog=vals,
                              groups=names,
                              alpha=0.05)
      print(tukey)
      means=np.round([np.mean(set1[parameter[t]]),np.mean(set2[parameter[t]]),np.mean(set3[parameter[t]]),np.mean(set4[parameter[t]])],decimals=2)
      stds=np.round([np.std(set1[parameter[t]]),np.std(set2[parameter[t]]),np.std(set3[parameter[t]]),np.std(set4[parameter[t]])],decimals=2)
      means_stds=[]
      for i in range(0,len(means)):
          means_stds.append(str(means[i])+'±'+str(stds[i]))
      nums=[len(set1[parameter[t]]),len(set2[parameter[t]]),len(set3[parameter[t]]),len(set4[parameter[t]])]
      
      #violin plot
      pal=sns.color_palette()
      palp=sns.color_palette("husl",8)
      cols=[palp[3],palp[5],palp[1],palp[7]]
      dpi=150
      fig,ax1=plt.subplots()
      sns.violinplot(data = data[['Condition',parameter[t]]], x=parameter[t], y="Condition", order=[Condition_1,Condition_2,Condition_3,Condition_4], palette=cols, showmeans=True,inner=None)
      ax1.set_xlabel(parameter[t])
      for i in range(len(means_stds)):
          ax1.annotate(str(means_stds[i]+'\nn='+str(nums[i])),xy=(means[i],i),horizontalalignment='center',verticalalignment='center')
      ax1.xaxis.set_label_position('top')
      ax1.xaxis.tick_top()
      plt.show()
      temp=parameter[t].split('(')[0]
      fig.savefig('filoTips_Output/Plots/'+temp+ '.tiff',dpi=dpi,bbox_inches='tight')


# ### Download filoTips locally

# In[9]:


#@title 8) Download filoTips_Output locally
os.chdir(path+'/filoTips analysis/filoTips_Output')
if not os.path.exists('Annotations'):
        os.makedirs('Annotations')
if not os.path.exists('Individual Experiments'):
        os.makedirs('Individual Experiments')
        
annot_files=glob.glob('*.tiff')
ind_files=glob.glob('*_.csv')

for i in ind_files:
  os.replace(i,'Individual Experiments/'+i)
for i in annot_files:
  os.replace(i,'Annotations/'+i)

os.chdir(path+'/filoTips analysis')

print('-- filoTips analysis complete --')

