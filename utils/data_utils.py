"""
# Data utility functions for training on the SUIM dataset
# Paper: https://arxiv.org/pdf/2004.01241.pdf  
"""
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import fnmatch
import itertools as it


def clown(mask):
    imw, imh = mask.shape[0], mask.shape[1]
    sal = np.zeros((imw, imh))
    for i in range(imw):
        for j in range(imh):
            if mask[i,j]==1 :
                sal[i, j] = 1 
            else: pass
    return np.expand_dims(sal, axis=-1) 

def processclownfish(img, mask, sal=False):
    # scaling image data and masks
    img = img / 255
    m = []
    for i in range(mask.shape[0]):
        m.append(clown(mask[i]))
    m = np.array(m)
    return (img, m)


def trainDataGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                    mask_color_mode="grayscale", target_size=(256,256), sal=False):
    # data generator function for driving the training
    image_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = None,
        save_prefix  = None,
        seed=1)
    # mask generator function for corresponding ground truth
    mask_datagen = ImageDataGenerator(**aug_dict)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = None,
        save_prefix  = None,
        seed = 1)
    # make pairs and return
    for (img, mask) in zip(image_generator, mask_generator):
        img, mask_indiv = processclownfish(img, mask, sal)
        
        yield (img, mask_indiv)


def valDataGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                    mask_color_mode="grayscale", target_size=(256,256), sal=False):
    # data generator function for driving the training
    valimage_datagen = ImageDataGenerator(**aug_dict)
    valimage_generator = valimage_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = 1,
        save_to_dir = None,
        save_prefix  = None,
        seed=1)
    # mask generator function for corresponding ground truth
    valmask_datagen = ImageDataGenerator(**aug_dict)
    valmask_generator = valmask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = 1,
        save_to_dir = None,
        save_prefix  = None,
        seed = 1)
    # make pairs and return
    for (img, mask) in zip(valimage_generator, valmask_generator):
        img, mask_indiv = processclownfish(img, mask, sal)
        yield (img, mask_indiv)


def getPaths(data_dir):
    # read image files from directory
    exts = ['*.png','*.PNG','*.jpg','*.JPG', '*.JPEG', '*.bmp']
    image_paths = []
    for pattern in exts:
        for d, s, fList in os.walk(data_dir):
            for filename in fList:
                if (fnmatch.fnmatch(filename, pattern)):
                    fname_ = os.path.join(d,filename)
                    print(fname_)
                    image_paths.append(fname_)
    return image_paths


