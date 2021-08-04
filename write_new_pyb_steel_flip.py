import numpy as np
import pickle
import glob
from random import shuffle

fn = f"STEEL/split_300_300_new.pyb"
with open(f"splits/{fn}", "rb") as f:
    train_images, test_images, validation_images = pickle.load(f)
defect_list = glob.glob(r".\datasets\STEEL\split_300_defects\*.jpg")
defect_list_name = []
for defect in defect_list:
    defect_list_name.append(defect.split("\\")[-1])
validation_images
dst_path = ".\splits\STEEL\split_300_300_flip_defect.pyb"

train_paths = [r".\datasets\STEEL\split_300_all",
               r".\datasets\STEEL\split_300_all_180",
               r".\datasets\STEEL\split_300_all_horizontal",
               r".\datasets\STEEL\split_300_all_vertical"]

original_split_path = ".\\datasets\\STEEL\\train_images\\"

train_names = []
train_original_names =[row[0] for row in train_images]
j = 0
for i, train_path in enumerate(train_paths):
    if i == 0 :
        image_names = glob.glob(train_path + "\*[!_GT].jpg")
        for image_name in image_names:
            name = image_name.split("\\")[-1]

            original_name = original_split_path + name
            index = train_original_names.index(original_name)
            is_fully_labeled = train_images[index][1]
            train_names.append((image_name, is_fully_labeled))
    else:
        image_names = glob.glob(train_path + "\*[!_GT].jpg")
        for image_name in image_names:
            name = image_name.split("\\")[-1]

            if name in defect_list_name:
                j+=1
                print(j)
                original_name = original_split_path + name
                index = train_original_names.index(original_name)
                is_fully_labeled = train_images[index][1]
                train_names.append((image_name, is_fully_labeled))

shuffle(train_names)
# Store data (serialize)
with open(dst_path, 'wb') as handle:
    pickle.dump((train_names, test_images, validation_images), handle, protocol=pickle.HIGHEST_PROTOCOL)
