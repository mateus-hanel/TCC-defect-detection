import numpy as np
from pickle5 import pickle
import glob
from random import shuffle

fn = f"STEEL/split_300_0_new.pyb"
with open(f"splits/{fn}", "rb") as f:
    train_images, test_images, validation_images = pickle.load(f)

dst_path = ".\splits\STEEL\split_300_0_flip.pyb"

train_paths = [r".\datasets\STEEL\split_300_all",
               r".\datasets\STEEL\split_300_all_180",
               r".\datasets\STEEL\split_300_all_horizontal",
               r".\datasets\STEEL\split_300_all_vertical"]

original_split_path = ".\\datasets\\STEEL\\train_images\\"

train_names = []
train_original_names =[row[0] for row in train_images]
for i, train_path in enumerate(train_paths):

    image_names = glob.glob(train_path + "\*[!_GT].jpg")
    for image_name in image_names:
        print(image_name)
        name = image_name.split("\\")[-1]
        original_name = original_split_path + name
        index = train_original_names.index(original_name)
        is_fully_labeled = train_images[index][1]
        train_names.append((image_name, is_fully_labeled))


shuffle(train_names)
# Store data (serialize)
with open(dst_path, 'wb') as handle:
    pickle.dump((train_names, test_images, validation_images), handle, protocol=pickle.HIGHEST_PROTOCOL)
