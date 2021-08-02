import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
import cv2
import pickle5 as pickle
import glob
from random import shuffle
import os

# Reads the split
'''
fn = f"STEEL/split_300_150_new.pyb"
with open(f"splits/{fn}", "rb") as f:
    train_images, test_images, validation_images = pickle.load(f)

train_images_filtered = [i for i in train_images if i[1]]
len(train_images)-len(train_images_filtered)
'''

def rle_to_mask(rle, image_size):
    if len(rle) % 2 != 0:
        raise Exception('Suspicious')

    w, h = image_size
    mask_label = np.zeros(w * h, dtype=np.float32)

    positions = rle[0::2]
    length = rle[1::2]
    for pos, le in zip(positions, length):
        mask_label[pos - 1:pos + le - 1] = 1
    mask = np.reshape(mask_label, (h, w), order='F')
    return mask

def read_annotations(fn):
    arr = np.array(pd.read_csv(fn), dtype=np.object)
    annotations_dict = {}
    for sample, _, rle in arr:
        #img_name = sample[:-4]
        img_name = sample

        annotations_dict[img_name] = rle

    return annotations_dict

def rle_encoding(x, normalized):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    if normalized:
        dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    else:
        dots = np.where(x.T.flatten()==255)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

annotations = read_annotations(r".\datasets\STEEL\train.csv")
len(annotations)
image_size = (1600, 256)
dst = r".\datasets\STEEL\masks"

plt.imshow(im)

for key, value in annotations.items():
    print(key)
    rle = list(map(int,value.split(" ")))
    mask = (rle_to_mask(rle, image_size)*255).astype(np.uint8)
    im = Image.fromarray(mask)
    break
    #im.save(key[:-4] + "_GT.png")

mask_rle = rle_to_mask(rle, image_size)
test = rle_encoding(mask_rle)
mask_2 = rle_to_mask(test,image_size)
rle
key
plt.imshow(mask_2)
img = cv2.imread(key[:-4] + "_GT.png")
plt.imshow(img)
img
rle_img = rle_encoding(img,False)
mask_rle = rle_to_mask(rle_img, image_size)
plt.imshow(mask_rle)
