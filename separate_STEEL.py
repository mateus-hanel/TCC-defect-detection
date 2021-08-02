
import cv2
import glob
from matplotlib import pyplot as plt
from pickle5 import pickle
from shutil import copyfile
import os.path

fn = f"STEEL/split_300_300_new.pyb"

with open(f"splits/{fn}", "rb") as f:
    train_images, test_images, validation_images = pickle.load(f)
len(train_images)
train_images[0]
dst = ".\\datasets\\STEEL\\split_300_good\\"
for image in train_images:
    name = image[0].split("\\")[-1]
    # Searches for the GT
    GT = image[0][:-4] + "_GT.png"
    if not os.path.isfile(GT):
            #dst_GT = dst + name[:-4] + "_GT.png"
            #copyfile(GT, dst_GT)

            dst_image = dst + name

            copyfile(image[0], dst_image)
    #break
