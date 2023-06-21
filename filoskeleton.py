#!/usr/bin/env python

#@title 1) Provide information describing the data and load ZeroCostDL4Mic requirements (code modified slightly from "U-Net_2D_Multilabel" notebook)
#code from ZeroCostDL4Mic 1.1
from __future__ import print_function

#get filename and micron/pixel ratio
from tkinter import Tk
from tkinter.filedialog import askdirectory
path = askdirectory(title='Select Folder') # shows dialog box and return the path
print('')
print(path+' selected as file path')

#updated inputs
um_per_pixel = None
while type(um_per_pixel) != float:
    try:
        um_per_pixel = float(input('Micron/Pixel Ratio: '))
        break
    except ValueError:
        print("Please enter a valid number...")

actin_channel = str(input('CELL BODY and filopodia STALK image extension. Must be found in all body and stalk image names: '))
filo_channel = str(input('FILOPODIA TIP image extension. Must be found in all filopodia tip image names: '))
RGB_insurance = input('Please make sure you also have an RGB image with CELL BODY, STALK, and TIPS combined into one. These images should have the extension "_RGB". Press enter only if you have channel-combined RGB images: ')
filo_name = str(input('Name of filopodia tip marker, (ex. Myo10): '))

#what models should be used?
#cell and stalk model
model_type = None
while model_type not in {"Default", "Custom"}:
    model_type = input('CELL BODY and STALK model type, "Default" or "Custom": ')
    
    if model_type in {"Default", "Custom"}:
        if model_type == 'Default':
            use_default_cell_model = True
        if model_type == 'Custom':
            use_default_cell_model = False
            custom_cell_model_DriveLink= input('Please copy and paste the Google DriveLink associated with your custom CELL BODY and STALK model: ')
    else:
        print('Please type "Default" or "Custom" verbatim to note which model you want to use.')
#filopodia tip model
model_type = None
while model_type not in {"Default", "Custom"}:
    model_type = input('FILOPODIA TIP model type, "Default" or "Custom": ')
    
    if model_type in {"Default", "Custom"}:
        if model_type == 'Default':
            use_default_filotip_model = True
        if model_type == 'Custom':
            use_default_filotip_model = False
            custom_filotip_model_DriveLink= input('Please copy and paste the Google DriveLink associated with your custom FILOPODIA TIP model: ')
    else:
        print('Please type "Default" or "Custom" verbatim to note which model you want to use.')
        
# do you want to perform a comparative_analysis        
comparative_analysis = None
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
while type(Annotation_DPI) != int:
    try:
        Annotation_DPI = int(input('Figure DPI (ex. "150","300","900"): '))
        break
    except ValueError:
        print("Please enter a valid number...")

#Calculate pixel_micron from micron_pixel
pixel_micron=1/float(um_per_pixel)

print('Inputs accepted. Initiating filoSkeleton analysis...')


import os
import subprocess
import sys
import imagecodecs
from builtins import any as b_any

os.chdir(path)
if not os.path.exists('filoSkeleton analysis'):
  os.makedirs('filoSkeleton analysis')

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install('data')
install('fpdf')
subprocess.run('pip install h5py==2.10',shell=True)
install('imagecodecs')

#code from ZeroCostDL4Mic 1.3
Notebook_version = '1.13'
Network = 'U-Net (2D) multilabel'


def filter_files(file_list, filter_list):
    filtered_list = []
    for fname in file_list:
        if b_any(fname.split('==')[0] in s for s in filter_list):
            filtered_list.append(fname)
    return filtered_list

def get_requirements_path():
    # Store requirements file in 'contents' directory 
    #current_dir = os.getcwd()
    #dir_count = current_dir.count('/') - 1
    #path = './' * (dir_count) + 'requirements.txt'
    req_path = path+'/filoSkeleton analysis/requirements.txt'
    return req_path

def build_requirements_file(before, after):
    req_path = get_requirements_path()

    # Exporting requirements.txt for local run
    os.chdir(path+'/filoSkeleton analysis')
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
from keras import backend as keras
from keras.callbacks import Callback

# General import
import numpy as np
import pandas as pd
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
  Saves all created patches in two new directories.

  Returns: - Two paths to where the patches are now saved
  """
  DEBUG = False

  Patch_source = os.path.join(path,'img_patches')
  Patch_target = os.path.join(path,'mask_patches')
  Patch_rejected = os.path.join(path,'rejected')

  #Here we save the patches, in the directory as they will not usually be needed after training
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


def save_augment(datagen,orig_img,dir_augmented_data=path+"/augment"):
  """
  Saves a subset of the augmented data for visualisation, by default.

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
  #patch_size = model.layers[0].output_shape[1:3]
  patch_size = model.layers[0].output_shape[0][1:3]

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
  exp_size = io.imread(path+'/TrainingDataExample_Unet2D.png').shape
  pdf.image(path+'/TrainingDataExample_Unet2D.png', x = 11, y = None, w = round(exp_size[1]/8), h = round(exp_size[0]/8))
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

#@title 2) Load filoSkeleton requirements
subprocess.call(['pip', 'install', '--upgrade', '--no-cache-dir', 'gdown'])
import os
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import math
import shutil
import tifffile as tiff
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns

install('researchpy')
import researchpy as rp





def get_distance(x2,x1,y2,y1):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

red,aqua,white,black,purple,gray1,gray2,gray3=(220,20,60),(127,255,212),(0,0,0),(255,255,255),(178,58,238),(128,128,128),(64,64,64),(32,32,32)
white,yellow,blue,green,pink,red,orange,black,light_orange,gray4=(255,255,255),(250,250,0),(51,153,255),(0,204,0),(255,0,127),(255,51,51),(153,76,0),(0,0,0),(255,178,102),(160,160,160)
#orange,purple=(255,125,64),(191,62,255)
pink,beige,peacock,peachpuff1=(255,130,171),(245,245,220),(51,161,201),(161,161,161)
random_color=[(222,184,135),(255,211,155),(238,197,145),(205,170,125),(139,115,85),(138,54,15),(138,51,36),(95,158,160),(152,245,255),(142,229,238)]

#@title 3) Download the deep learning agent and prepare for predictions
os.chdir(path+'/filoSkeleton analysis')

if not os.path.exists(path+'/filoSkeleton analysis/filoSkeleton cell predictions'):
  os.makedirs(path+'/filoSkeleton analysis/filoSkeleton cell predictions')
if not os.path.exists(path+'/filoSkeleton analysis/filoSkeleton filopodia tip predictions'):
  os.makedirs(path+'/filoSkeleton analysis/filoSkeleton filopodia tip predictions')
if not os.path.exists(path+'/filoSkeleton analysis/filoSkeleton cell source'):
  os.makedirs(path+'/filoSkeleton analysis/filoSkeleton cell source')
if not os.path.exists(path+'/filoSkeleton analysis/filoSkeleton filopodia tip source'):
  os.makedirs(path+'/filoSkeleton analysis/filoSkeleton filopodia tip source')
if not os.path.exists(path+'/filoSkeleton analysis/filoSkeleton merged'):
  os.makedirs(path+'/filoSkeleton analysis/filoSkeleton merged')

os.chdir(path)
green_files=sorted(glob.glob('*'+actin_channel+'.tiff'))
if len(green_files)==0:
  green_files=sorted(glob.glob('*'+actin_channel+'.tif'))
for file in green_files:
  shutil.copy(file,path+'/filoSkeleton analysis/filoSkeleton cell source/'+file)

red_files=sorted(glob.glob('*'+filo_channel+'.tiff'))
if len(red_files)==0:
  red_files=sorted(glob.glob('*'+filo_channel+'.tif'))
for file in red_files:
  shutil.copy(file,path+'/filoSkeleton analysis/filoSkeleton filopodia tip source/'+file)

merge_files=sorted(glob.glob('*RGB.tiff'))
if len(merge_files)==0:
  merge_files=sorted(glob.glob('*RGB.tif'))
for file in merge_files:
  shutil.copy(file,path+'/filoSkeleton analysis/filoSkeleton merged/'+file)

#Sandra_Final
os.chdir(path+'/filoSkeleton analysis')
if use_default_cell_model == True:
  subprocess.call(['gdown', '--id', '1wnCJAVDNxd9pSpUc6KgXppig3N-MKExD'])
  subprocess.call(['tar', '-xf', 'Sandra_Final.zip'])

if use_default_cell_model ==False:
  subprocess.run(['gdown', custom_cell_model_DriveLink])
  zip1=glob.glob('*.zip')[0]
  custom_cell_model_name = zip1.replace('.zip','')
  subprocess.run(['tar', '-xf', zip1])
  
  
#Jerry_Final_Myo10
if use_default_filotip_model ==True:
  subprocess.call(['gdown', '--id', '1UFCtMLfV8PP1eX2imUa7aQg0HRbtlKcw'])
  subprocess.call(['tar', '-xf', 'Jerry_Final_Myo10.zip'])
  
if use_default_filotip_model==False:  
  subprocess.run(['gdown', custom_filotip_model_DriveLink])
  zip2=glob.glob('*.zip')[0]
  custom_filotip_model_name = zip2.replace('.zip','')
  subprocess.run(['tar', '-xf', zip2])

#make images 8-bit
#os.chdir(path+'/filoSkeleton cell source')
#for file in green_files:
#  img=cv2.imread(file,-1)
#  img=cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#  tiff.imsave(file,img)
#os.chdir(path+'/filoSkeleton filopodia tip source')
#for file in red_files:
#  img=cv2.imread(file,-1)
#  img=cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#  tiff.imsave(file,img)

os.chdir(path)

#@title 4) Have the model generate masks of cell bodies, filopodia, and background


# ------------- Initial user input ------------
Data_folder = [path+'/filoSkeleton analysis/filoSkeleton cell source',path+'/filoSkeleton analysis/filoSkeleton filopodia tip source']
Results_folder = [path+'/filoSkeleton analysis/filoSkeleton cell predictions',path+'/filoSkeleton analysis/filoSkeleton filopodia tip predictions']
Prediction_model_folder = [path+"/filoSkeleton analysis/Sandra_Final",path+'/filoSkeleton analysis/Jerry_Final_Myo10']
if use_default_cell_model==False:
  Prediction_model_folder[0]= path+'/filoSkeleton analysis/'+custom_cell_model_name
if use_default_filotip_model==False:
  Prediction_model_folder[1]=path+'/filoSkeleton analysis/'+custom_filotip_model_name

for y in range(0,len(Data_folder)):
  Use_the_current_trained_model = False

  #Here we find the loaded model name and parent path
  Prediction_model_name = os.path.basename(Prediction_model_folder[y])
  Prediction_model_path = os.path.dirname(Prediction_model_folder[y])


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
  source_dir_list = os.listdir(Data_folder[y])
  number_of_dataset = len(source_dir_list)
  print('Number of dataset found in the folder: '+str(number_of_dataset))

  predictions = []
  for i in range(number_of_dataset):
    predictions.append(predict_as_tiles(os.path.join(Data_folder[y], source_dir_list[i]), unet))
    #predictions.append(prediction(os.path.join(Data_folder[y], source_dir_list[i]), os.path.join(Prediction_model_path, Prediction_model_name)))


  # Save the results in the folder along with the masks according to the set threshold
  saveResult(Results_folder[y], predictions, source_dir_list, prefix=prediction_prefix)


  # ------------- For display ------------
  print('--------------------------------------------------------------')
#  os.chdir(Results_folder[y])
#  files=sorted(glob.glob('*.tif'))
  #for file in files:
  #  name=file.replace('.tif','.tiff')
  #  os.rename(file,name)
#  os.chdir(path)

#  def show_prediction_mask(file=os.listdir(Data_folder[y])):
#
#    plt.figure(figsize=(10,6))
#    # Wide-field
#    plt.subplot(1,2,1)
#    plt.axis('off')
#    img_Source = plt.imread(os.path.join(Data_folder[y], file))
#    plt.imshow(img_Source, cmap='gray')
#    plt.title('Source image',fontsize=15)
#    # Prediction
#    plt.subplot(1,2,2)
#    plt.axis('off')
#    img_Prediction = plt.imread(os.path.join(Results_folder[y], prediction_prefix+file))
#    plt.imshow(img_Prediction, cmap='gray')
#    plt.title('Prediction',fontsize=15)

#  interact(show_prediction_mask);

#@title 5) Rename and organize predictions
os.chdir(path+'/filoSkeleton analysis/filoSkeleton cell predictions')
names=sorted(glob.glob('*'+actin_channel+'.tif'))
actin_channels,filo_channels,merged_names=[],[],[]
for i in names:
  temp=i.replace('Predicted_','')
  actin_channels.append(temp)
  os.rename(i,temp)
  merged_names.append(temp.replace(actin_channel+'.tif','RGB.tif'))

os.chdir(path+'/filoSkeleton analysis/filoSkeleton filopodia tip predictions')
names=sorted(glob.glob('Predicted_*'))
for i in names:
  temp=i.replace('Predicted_','')
  os.rename(i,temp)

names=sorted(glob.glob('*'+filo_channel+'.tif'))
for i in names:
  filo_channels.append(i)

#@title 6) Use the prediction masks to quantify filopodia
print('--Analyzing images--')
for q in range(0,len(actin_channels)):
    print('--Analyzing '+str(q+1)+' / '+str(len(actin_channels))+' images--')
    os.chdir(path+'/filoSkeleton analysis/filoSkeleton cell source')
    img=cv2.imread(actin_channels[q],-1)
    os.chdir(path+'/filoSkeleton analysis/filoSkeleton merged')
    img_show=cv2.imread(merged_names[q])
    img_show=cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
    cols, rows = img_show.shape[0],img_show.shape[1]
    brightness = np.sum(img_show) / (255 * cols * rows)
    minimum_brightness = 0.10
    ratio = brightness / minimum_brightness
    img_show= cv2.convertScaleAbs(img_show, alpha = 1 / ratio, beta = 0)

    img_show1=cv2.imread(merged_names[q])
    img_show1=cv2.cvtColor(img_show1, cv2.COLOR_BGR2RGB)
    cols, rows = img_show1.shape[0],img_show1.shape[1]
    brightness = np.sum(img_show1) / (255 * cols * rows)
    minimum_brightness = 0.50
    ratio = brightness / minimum_brightness
    img_show1= cv2.convertScaleAbs(img_show1, alpha = 1 / ratio, beta = 0)

    os.chdir(path+'/filoSkeleton analysis/filoSkeleton cell source')
    img_copy=cv2.imread(actin_channels[q])
    img_copy[np.where(img_copy>0)]=0
    img_copy2=np.copy(img_copy)
    img_copy3=np.copy(img_copy)
    img_copy4=np.copy(img_copy)
    img_copy5=np.copy(img_copy)
    stalk_img1=cv2.imread(actin_channels[q])
    filo_img1=cv2.imread(actin_channels[q])
    filo_img1[np.where(filo_img1>0)]=0
    filo_img2=np.copy(filo_img1)
    #red_img=cv2.imread(filo_channels[q],-1)

    #read in masks
    os.chdir(path+'/filoSkeleton analysis/filoSkeleton cell predictions')
    body_stalk=cv2.imread(actin_channels[q],-1)
    body,stalk=np.copy(body_stalk),np.copy(body_stalk)
    body[np.where(body!=1)]=0
    stalk[np.where(body_stalk!=2)]=0
    kernel = np.ones((7,7), np.uint8)
    stalk=cv2.dilate(stalk,kernel)
    os.chdir(path+'/filoSkeleton analysis/filoSkeleton filopodia tip predictions')
    filo=cv2.imread(filo_channels[q],-1)
    os.chdir(path+'/filoSkeleton analysis/filoSkeleton filopodia tip source')
    filo_int=cv2.imread(filo_channels[q],-1)

    #get cell contours
    contours, hierarchy = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    centroids,areas,perimeters,aspect_ratios,circularity,body_means=[],[],[],[],[],[]
    for i in range(0,len(contours)):
        M=cv2.moments(contours[i])
        if M['m00']!=0:
            area=cv2.contourArea(contours[i])/(pixel_micron**2)
            if area>25:
                centroid=(int(M['m10']/M['m00']),int(M['m01']/M['m00']))
                centroids.append(centroid)
                area=cv2.contourArea(contours[i])/(pixel_micron**2)
                areas.append(area)
                img_copy5=cv2.drawContours(img_copy5,contours,i,(147,147,147),-1)
                cv2.drawContours(img_copy5,contours,i,(0,0,0,),5)
                body_val=filo_int[np.where((img_copy5==list((147,147,147))).all(axis=2))]
                body_vals=[]
                for val in body_val:
                  vals=[str(val)]
                  if len(vals)==1:
                    body_vals.append(val)
                  else:
                    body_vals.append(val[2])
                body_means.append(np.mean(body_vals))
                perimeter=cv2.arcLength(contours[i],True)/(pixel_micron)
                perimeters.append(perimeter)
                rect=cv2.minAreaRect(contours[i])
                wh=rect[1]
                w=np.min(wh)
                h=np.max(wh)
                aspect_ratios.append(float(w)/h)
                circularity.append((4*math.pi*area)/(perimeter**2))
                img_copy2=cv2.drawContours(np.copy(img_copy),contours,i,white,-1)
                img_show=cv2.drawContours(img_show,contours,i,aqua,-1)
                img_copy2=cv2.drawContours(img_copy2,contours,i,white,3)
                img_copy3=cv2.drawContours(img_copy3,contours,i,aqua,1)

    #get all filo contours
    filo[np.where((img_copy2==list(white)).all(axis=2))]=0
    all_filo=np.copy(filo)

    contours, hierarchy = cv2.findContours(all_filo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filo_centroids,filo1_tip_body_ratio,filo1_nums,filo1_len,filo1_cent,filo1_coord,filo1_cell,filo1_val,filo1_cell_intensity=[],[],[],[],[],[],[],[],[]
    c=1
    for i in range(0,len(contours)):
        M=cv2.moments(contours[i])
        if M['m00']!=0:
          filo_centroid=(int(M['m10']/M['m00']),int(M['m01']/M['m00']))
          filo_centroids.append(centroid)
          cv2.drawContours(filo_img1,contours,i,red,-1)
          pts = np.where((filo_img1==list(red)).all(axis=2))
          intensity1=(filo[pts[0],pts[1]])
          intensity2=(stalk[pts[0],pts[1]])
          x1,y1=filo_centroid[0],filo_centroid[1]
          body_pts=np.where((img_copy3==list(aqua)).all(axis=2))
          x2,y2=body_pts[1],body_pts[0]
          dist=[]
          for j in range(len(x2)):
            dist.append(get_distance(x2[j],x1,y2[j],y1))
          shortest_len=dist[np.argmin(dist)]/pixel_micron
          shortest_cent=x2[np.argmin(dist)],y2[np.argmin(dist)]
          x1,y1=shortest_cent
          dist=[]
          for j in range(len(centroids)):
            x2,y2=centroids[j]
            dist.append(get_distance(x2,x1,y2,y1))
          closest_cell=np.argmin(dist)

          if 1 in intensity1 and 2 in intensity2:
            cv2.drawContours(img_show,contours,i,red,-1)
            cv2.line(img_show,filo_centroid,shortest_cent,yellow,3)
            filo1_len.append(shortest_len)
            filo1_cent.append(shortest_cent)
            filo1_cell.append(closest_cell+1)
            filo1_cell_intensity.append(body_means[closest_cell])
            reverse_coord=(filo_centroid[1],filo_centroid[0])
            if type(filo_int[reverse_coord])=='list':
              filo1_val.append(str(filo_int[reverse_coord][2]))
              filo1_tip_body_ratio.append(str(filo_int[reverse_coord][2]/body_means[closest_cell]))
            else:
              filo1_val.append(str(filo_int[reverse_coord]))
              filo1_tip_body_ratio.append(str(filo_int[reverse_coord]/body_means[closest_cell]))
            filo1_coord.append(reverse_coord)
          cv2.drawContours(filo_img1,contours,i,gray3,-1)
          c=c+1

    #summarize filo information
    rootName = actin_channels[q].replace(actin_channel+'.tif','')
    filo1_protein=np.repeat(filo_name,len(filo1_len))
    filo_names=np.repeat(rootName,len(filo1_len))
    avg_len=np.mean(filo1_len)

    #filo_output dataframe
    filo1_protein=np.repeat(filo_name,len(filo1_len))
    filo_names=np.repeat(rootName,len(filo1_len))



    filo1_={'Experiment Name':filo_names,'Cell Assignment':filo1_cell,'Cell Body Average Intensity':filo1_cell_intensity,'Length (um)':filo1_len,'Filo Tip Centroid Coordinates':filo1_coord,'Filo Tip Centroid Intensity':filo1_val,'Filo Tip/Cell Body Ratio':filo1_tip_body_ratio,'Filopodia Proteins Present':filo1_protein}
    filo_output=pd.DataFrame(filo1_)

    #summarize and prepare for result outputs
    filo1_cell_ind=np.asarray(filo1_cell)
    final1_2,final1,final2,finalprojection=[],[],[],[]
    for i in range(1,len(centroids)+1):
      final1.append(len(np.where(filo1_cell_ind==i)[0]))
    cell_num=list(range(1,len(centroids)+1))

    filos1_2_micron=[i/j for i,j in zip(final1_2,perimeters)]
    filos1_micron=[i/j for i,j in zip(final1,perimeters)]
    filos2_micron=[i/j for i,j in zip(final2,perimeters)]
    projection_micron=[i/j for i,j in zip(finalprojection,perimeters)]

    #combine lists into a pandas df and export
    os.chdir(path+'/filoSkeleton analysis')
    if not os.path.exists('filoSkeleton output'):
        os.makedirs('filoSkeleton output')
    exp_num_cell=np.repeat(rootName,len(centroids))
    cell_num=[]
    for i in range(0,len(centroids)):
      cell_num.append(i+1)

    #calculate filopodia normalized to perimeter
    filos_perimeter = [i/j for i,j in zip(final1,perimeters)]

    #finish combining lists
    cell_dict={'Experiment Name':exp_num_cell,'Cell Number':cell_num,'Aspect Ratio':aspect_ratios,'Circularity':circularity,'Average Body Intensity':body_means,'Filos/Cell ('+filo_name+')':final1,'Filos/Perimeter (Filos/micron)':filos_perimeter, 'Avg Filo Length (um)':avg_len,'Cell Area (um^2)':areas,'Perimeter (um)':perimeters}
    cell_output=pd.DataFrame(cell_dict)
    cell_output.to_csv('filoSkeleton output/'+str(rootName.replace(actin_channel+'.tif',''))+'_Cell_Output_.csv',index=False)
    filo_output.to_csv('filoSkeleton output/'+str(rootName.replace(actin_channel+'.tif',''))+'_Filo_Output_.csv',index=False)

    #make annotation
    fig, final=plt.subplots(1,2)
    final[0].imshow(img_show1)
    final[0].set_title('Merged')
    final[0].axis('off')
    final[1].imshow(img_show)
    final[1].set_title('filoSkeleton')
    final[1].axis('off')
    plt.savefig('filoSkeleton output/'+str(rootName.replace(actin_channel+'.tif',''))+'_Annotation.tiff',dpi=Annotation_DPI)

#@title 7) Pool together the analysis
os.chdir(path+'/filoSkeleton analysis')
cell_outputs=sorted(glob.glob("filoSkeleton output/*Cell_Output_.csv"))
filo_outputs=sorted(glob.glob('filoSkeleton output/*Filo_Output_.csv'))
cell_dfs = (pd.read_csv(f) for f in cell_outputs)
cell_dfs = pd.concat(cell_dfs, ignore_index=True)
filo_dfs = (pd.read_csv(f) for f in filo_outputs)
filo_dfs = pd.concat(filo_dfs, ignore_index=True)
cell_dfs.to_csv('filoSkeleton output/Total_Cell_Output.csv',index=False)
filo_dfs.to_csv('filoSkeleton output/Total_Filo_Output.csv',index=False)

#@title 8) Optional data visualization
cell_file=pd.read_csv(path+'/filoSkeleton analysis/filoSkeleton output/Total_Cell_Output.csv')
if not os.path.exists(path+'/filoSkeleton analysis/filoSkeleton output/Plots'):
        os.makedirs(path+'/filoSkeleton analysis/filoSkeleton output/Plots')

if comparative_analysis ==True:
  prompt1=[Condition_1,Condition_2,Condition_3,Condition_4]
  if '' in prompt1:
    prompt1.remove('')
    if '' in prompt1:
      prompt1.remove('')
  prompt1=str(len(prompt1))
  parameter=['Aspect Ratio','Circularity','Filos/Cell ('+filo_name+')','Filos/Perimeter (Filos/micron)','Avg Filo Length (um)','Cell Area (um^2)','Perimeter (um)']

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
      set1=data.loc[data['Condition'].str.contains(Condition_1)].reset_index(drop=True)
      set2=data.loc[data['Condition'].str.contains(Condition_2)].reset_index(drop=True)

      print('')
      print('--'+parameter[t]+'--')
      print('--Mann-Whitney U Test--')
      kruskal=stats.mannwhitneyu(set1[parameter[t]],set2[parameter[t]])
      print(kruskal)
      print(' ')

      #t-test
      print('--'+parameter[t]+'--')
      print('--T-test--')
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
      temp=parameter[t].split('(')[0]
      temp=temp.replace('/','_')
      fig.savefig(path+'/filoSkeleton analysis/filoSkeleton output/Plots/'+temp+ '.tiff',dpi=dpi,bbox_inches='tight')

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
      set1=data.loc[data['Condition'].str.contains(Condition_1)].reset_index(drop=True)
      set2=data.loc[data['Condition'].str.contains(Condition_2)].reset_index(drop=True)
      set3=data.loc[data['Condition'].str.contains(Condition_3)].reset_index(drop=True)

      #kruskal-wallis
      print('')
      print('--'+parameter[t]+'--')
      print('-----Kruskal-Wallis Test-----')
      kruskal=stats.kruskal(set1[parameter[t]],set2[parameter[t]],set3[parameter[t]])
      print(kruskal)
      print(' ')

      #ANOVA
      print('--'+parameter[t]+'--')
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
      temp=parameter[t].split('(')[0]
      temp=temp.replace('/','_')
      fig.savefig(path+'/filoSkeleton analysis/filoSkeleton output/Plots/'+temp+ '.tiff',dpi=dpi,bbox_inches='tight')


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
      set1=data.loc[data['Condition'].str.contains(Condition_1)].reset_index(drop=True)
      set2=data.loc[data['Condition'].str.contains(Condition_2)].reset_index(drop=True)
      set3=data.loc[data['Condition'].str.contains(Condition_3)].reset_index(drop=True)
      set4=data.loc[data['Condition'].str.contains(Condition_4)].reset_index(drop=True)

      #kruskal-wallis
      print('')
      print('--'+parameter[t]+'--')
      print('-----Kruskal-Wallis Test-----')
      kruskal=stats.kruskal(set1[parameter[t]],set2[parameter[t]],set3[parameter[t]],set4[parameter[t]])
      print(kruskal)
      print(' ')

      #ANOVA
      print('--'+parameter[t]+'--')
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
      temp=parameter[t].split('(')[0]
      temp=temp.replace('/','_')
      fig.savefig(path+'/filoSkeleton analysis/filoSkeleton output/Plots/'+temp+ '.tiff',dpi=dpi,bbox_inches='tight')

#@title 9) Organize files
os.chdir(path+'/filoSkeleton analysis/filoSkeleton output')
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

print('--filoSkeleton analysis complete--')
