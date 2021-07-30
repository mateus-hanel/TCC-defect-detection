import os
import numpy as np
import glob
import pickle
from shutil import copyfile

src = r'.\datasets\KSDD2\train_defects'
dst = r'.\datasets\KSDD2\train_defects_16'

fn = f"KSDD2/split_16.pyb"
with open(f"splits/{fn}", "rb") as f:
    train_samples, test_samples = pickle.load(f)
train_original_names =[row[0] for row in train_samples]


image_names = glob.glob(src + "\*[!_GT].png")

for image_name in image_names:
    image = image_name.split("\\")[-1][:-4]
    name = int(image)
    index = train_original_names.index(name)
    is_fully_labeled = train_samples[index][1]

    if is_fully_labeled:
        label = image_name[:-4] + "_GT.png"
        image_dst = dst + "\\" + image + ".png"
        label_dst = dst + "\\" + image + "_GT.png"
        #print(image_name, label)
        print(image_dst, label_dst)
        copyfile(image_name,image_dst)
        copyfile(label,label_dst)
