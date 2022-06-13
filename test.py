"""
# Test script for the UNet
    # for 5 object categories: HD, FV, RO, RI, WR 
# See https://arxiv.org/pdf/2004.01241.pdf  
"""
from __future__ import print_function, division
import os
import ntpath
import numpy as np
from PIL import Image
from os.path import join, exists
import cv2
from utils.data_utils import getPaths
from utils.measure_utils import db_eval_boundary, IoU_bin
from models.SURNet import RUSnet, RUSnet_vgg16, RUSnet_vgg19, RUSnet_resnet50
from config import args_setting

args = args_setting()
def read_and_bin(im_path):
    img = Image.open(im_path).resize((im_h, im_w))
    # print(img.size)
    img = np.array(img)/255.
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img
def resize(im_path):
    img = Image.open(im_path).resize((im_h, im_w))
    img = img.convert('L') 
    img = np.array(img)
    return img

def testGenerator(test_output_dir, im_w, im_h):
    # test all images in the directory
    assert exists(test_dir), "local image path doesnt exist"
    imgs = []
    for p in getPaths(test_dir):
        img = Image.open(p).resize((im_w, im_h))
        img = np.array(img)/255.
        img = np.expand_dims(img, axis=0)
        # inference
        out_img = model.predict(img)
        # thresholding
        out_img[out_img>0.5] = 1.
        out_img[out_img<=0.5] = 0.
        img_name = ntpath.basename(p).split('.')[0] + '.bmp'
        ROs = np.reshape(out_img[0,:,:,0], (im_h, im_w))
        img_arr = np.uint8(ROs*255.)
        result = cv2.erode(img_arr, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations = 1)
        result_dil = cv2.dilate(result, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations = 2)
        Image.fromarray(result_dil).save(test_output_dir+img_name)

test_dir = "../dataset/clownfish_v2/test_modify/img/"
samples_dir = "data/test/clownfish/"

obj_cat = args.model+ args.dataset_place + args.loss + str(args.dataset_num)+ '/'
test_output_dir = samples_dir + obj_cat
if not exists(test_output_dir): os.makedirs(test_output_dir)


im_res_ = (240, 320, 3)
if args.model == 'orimodel':
   model = RUSnet(im_res=im_res_, n_classes=1).model
elif args.model == 'vgg16':
    model = RUSnet_vgg16(im_res=im_res_, n_classes=1).model
elif args.model == 'vgg19':
    model = RUSnet_vgg19(im_res=im_res_, n_classes=1).model   
elif args.model == 'resnet50':
    model = RUSnet_resnet50(im_res=im_res_, n_classes=1).model   


im_h, im_w = im_res_[0], im_res_[1]


ckpt_dir = 'ckpt/'                                                                                       
ckpt_name = args.model+ args.dataset_place + args.loss + str(args.dataset_num)+".hdf5"

model.load_weights(ckpt_dir + ckpt_name)

# test images
testGenerator(test_output_dir, im_w, im_h)


# accumulate F1/iou values in the lists
Ps, Rs, F1s, IoUs = [], [], [], []
gen_paths = sorted(getPaths(test_output_dir))
real_paths = sorted(getPaths(real_mask_dir))
for gen_p, real_p in zip(gen_paths, real_paths):
    gen, real = read_and_bin(gen_p), resize(real_p)
    if (np.sum(real)>0):
        precision, recall, F1 = db_eval_boundary(real, gen)
        iou = IoU_bin(real, gen)
        # print ("{0}:>> P: {1}, R: {2}, F1: {3}, IoU: {4}".format(gen_p, precision, recall, F1, iou))
        Ps.append(precision) 
        Rs.append(recall)
        F1s.append(F1)
        IoUs.append(iou)

print ("Avg. F: {0}".format(100.0*np.mean(F1s)))
print ("Avg. IoU: {0}".format(100.0*np.mean(IoUs)))

