# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 20:19:35 2017

@author: SeongMin
"""

from glob import glob
from PIL import Image
import PIL
import numpy as np
# random 함수를 위해 import
import random
import os
#import matplotlib.pyplot as plt


DATA_AUG_TIMES = 1


def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


#src_dir = './data/Train400'
src_dir = './data/Train50'
#src_dir = './data/Train'

save_dir = './data'
pat_size = 40
stride = 14
step = 0

#bat_size = 128
bat_size = 128

# check output arguments
#from_file =  "./data/img_clean_pats.npy" 
from_file =  "./data/img_clean_pats.csv" 
num_pic = 10

#global DATA_AUG_TIMES

count = 0

# glob으로 file 목록 뽑음. (LIST 형태로)
filepaths = glob(src_dir + '/*.png') + glob(src_dir + '/*.bmp') + glob(src_dir + '/*.jpg')

print("number of training data %d" % len(filepaths))

scales = [1, 0.9, 0.8, 0.7]

# len(filepaths)는 400 즉, 400번 반복.
for i in range(len(filepaths)):
    img = Image.open(filepaths[i]).convert('L')  # convert RGB to gray
        
        #scale 수 만큼 반복.
    for s in range(len(scales)):
        # image scaling
        newsize = (int(img.size[0] * scales[s]), int(img.size[1] * scales[s]))
        img_s = img.resize(newsize, resample = PIL.Image.BICUBIC)  # do not change the original img
        im_h, im_w = img_s.size #im_h, im+w에는 scaling한 h,w가 저장됨.
            
#       if i == 10:
#       print(i)
#       plt.imshow(img_s)
        
        for x in range(0 + step, (im_h - pat_size + 2), stride):
            for y in range(0 + step, (im_w - pat_size + 2), stride):
                count += 1
                # 패치 개수 count하는 것
            
origin_patch_num = count * DATA_AUG_TIMES
#print(origin_patch_num) # 238400

#patch개수를 bat_size에 맞게 수정.
if origin_patch_num % bat_size != 0:
    numPatches = (origin_patch_num / bat_size + 1) * bat_size
    # print('들어오나 확인') #들어옴
else:
    numPatches = origin_patch_num

# numPatches를 integer로 바꾸기 위해 내가 넣은 코드
# 안바꿀경우 numPatches가 float형이라 오류가 뜸.
numPatches = int(numPatches)
inputs = np.zeros((numPatches, pat_size, pat_size, 1), dtype="uint8")

count = 0
# generate patches
# 400번 반복.
# filepaths 길이만큼 반복, 그 각각에서 scale계수의 수만큼 반복
# 그리고 스케일링.
for i in range(len(filepaths)):
    img = Image.open(filepaths[i]).convert('L')
    for s in range(len(scales)):
        newsize = (int(img.size[0] * scales[s]), int(img.size[1] * scales[s]))
        # print newsize
        img_s = img.resize(newsize, resample=PIL.Image.BICUBIC)
        img_s = np.reshape(np.array(img_s, dtype="uint8"),
                               (img_s.size[1], img_s.size[0], 1))  # extend one dimension
#        if i == 10:
#            plt.imshow(img_s)
        for j in range(DATA_AUG_TIMES): # patch의 계수만큼 반복.
            im_h, im_w, _ = img_s.shape 
            for x in range(0 + step, im_h - pat_size + 1, stride):
                for y in range(0 + step, im_w - pat_size + 1, stride):
                    inputs[count, :, :, :] = data_augmentation(img_s[x:x + pat_size, y:y + pat_size, :], \
                                                                   random.randint(0, 7))
                    count += 1


# pad the batch
if count < numPatches:
    to_pad = numPatches - count
    inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]
    
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
np.save(os.path.join(save_dir, "img_clean_pats"), inputs)
print("size of inputs tensor = " + str(inputs.shape))

#for i in range(2):
#        img = Image.open(filepaths[i]).convert('L')  # convert RGB to gray
#        #scale 수 만큼 반복.
#        for s in range(len(scales)):
#            # image scaling
#            newsize = (int(img.size[0] * scales[s]), int(img.size[1] * scales[s])) , scale된 size (180, 180) 즉, 튜플 형태로 size 출력.
#            print(newsize)

# calculate the number of patches




#def generate_patches(isDebug=False):
#    print('실행되나?')
