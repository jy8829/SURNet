"""
# Training pipeline of UNet on SUIM 
"""
from __future__ import print_function, division
import os
from os.path import join, exists
from keras import callbacks

from models.SURNet import RUSnet, RUSnet_vgg16, RUSnet_vgg19, RUSnet_resnet50

from utils.data_utils import trainDataGenerator
from utils.data_utils import valDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from config import args_setting
args = args_setting()

## dataset directory
dataset_name = "clownfish_v2"
# dataset_name = 'cavefish'
train_dir = "../dataset/clownfish_v2/dataset_split/" + args.dataset_place +'/'+  str(args.dataset_num)
# train_dir = "../dataset/Cavefish/train/"
val_dir = "../dataset/clownfish_v2/test_modify/"

ckpt_dir = 'ckpt/'                                                                                       

ckpt_name = args.model+ args.dataset_place + args.loss + str(args.dataset_num)+".hdf5"
# ckpt_name = 'cavefish.hdf5'
model_ckpt_name = join(ckpt_dir, ckpt_name)
if not exists(ckpt_dir): os.makedirs(ckpt_dir)

im_res_ = (240, 320, 3)
if args.model == 'orimodel':
   model = RUSnet(im_res=im_res_, n_classes=1).model
elif args.model == 'vgg16':
    model = RUSnet_vgg16(im_res=im_res_, n_classes=1).model
elif args.model == 'vgg19':
    model = RUSnet_vgg19(im_res=im_res_, n_classes=1).model   
elif args.model == 'resnet50':
    model = RUSnet_resnet50(im_res=im_res_, n_classes=1).model   


batch_size = args.batch_size
num_epochs = args.epochs
# setup data generator
train_data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

val_data_gen_args = dict(fill_mode='nearest')

model_checkpoint = callbacks.ModelCheckpoint(model_ckpt_name, 
                                   monitor = 'loss', 
                                   verbose = 1, mode= 'auto',
                                   save_weights_only = True,
                                   save_best_only = True)

# data generator
train_gen = trainDataGenerator(batch_size, # batch_size 
                            train_dir,# train-data dir
                            "img", # image_folder 
                            "mask", # mask_folder
                            train_data_gen_args, # aug_dict
                            image_color_mode="rgb", 
                            mask_color_mode="grayscale",
                            target_size = (im_res_[0], im_res_[1]))

val_gen = valDataGenerator(1, # batch_size 
                            val_dir,# train-data dir
                            "img", # image_folder 
                            "mask", # mask_folder
                            val_data_gen_args, # aug_dict
                            image_color_mode="rgb", 
                            mask_color_mode="grayscale",
                            target_size = (im_res_[0], im_res_[1]))

if args.dataset_place == 'mosaic':
    step =  int(args.dataset_num)*1.3/args.batch_size
    val_step =  (int(args.dataset_num)*1.3/args.batch_size)/2
else :
    step =  int(args.dataset_num)/args.batch_size
    val_step =  (int(args.dataset_num)/args.batch_size)/2

results = model.fit_generator(train_gen, 
                    steps_per_epoch = int(step),
                    validation_data = val_gen, 
                    validation_steps = int(val_step),
                    epochs = num_epochs,
                    callbacks = [model_checkpoint])   

acc = results.history['accuracy']
val_acc = results.history['val_accuracy']
loss = results.history['loss']
val_loss = results.history['val_loss']
    
epochs = range(len(acc))
    
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("log_acc")
plt.legend()
plt.savefig(ckpt_dir+args.dataset_place + '/' + args.model + args.dataset_place + args.loss + str(args.dataset_num) +'_acc.png')       
plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend()       
plt.savefig(ckpt_dir+args.dataset_place + '/' + args.model + args.dataset_place +args.loss + str(args.dataset_num) +'_loss.jpg')

