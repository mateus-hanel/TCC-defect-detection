import numpy as np
import pickle
import glob
from random import shuffle

fn = f"KSDD2/split_246.pyb"
with open(f"splits/{fn}", "rb") as f:
    train_samples, test_samples = pickle.load(f)
len(train_samples)
with open('defect_list.pyb', "rb") as f:
    defect_list= pickle.load(f)

dst_path = ".\splits\KSDD2\split_246_complete.pyb"

train_paths = [r".\datasets\KSDD2\train",
               r".\datasets\KSDD2\train_180",
               r".\datasets\KSDD2\train_flipped_horizontal",
               r".\datasets\KSDD2\train_flipped_vertical"]
train_paths_mask_aug = [r".\datasets\KSDD2\synthetic_train_normal_246"]

#train_paths = [r".\datasets\KSDD2\train"]

train_names = []
train_original_names =[row[0] for row in train_samples]

# Adds the flip part of the split
for i, train_path in enumerate(train_paths):
    if i == 0:
        image_names = glob.glob(train_path + "\*[!_GT].png")
        for image_name in image_names:
            name = int(image_name.split("\\")[-1][:-4])
            index = train_original_names.index(name)
            is_fully_labeled = train_samples[index][1]
            train_names.append((image_name, is_fully_labeled))
    else:
        image_names = glob.glob(train_path + "\*[!_GT].png")
        for image_name in image_names:
            name = int(image_name.split("\\")[-1][:-4])
            if str(name) in defect_list:
                index = train_original_names.index(name)
                is_fully_labeled = train_samples[index][1]
                train_names.append((image_name, is_fully_labeled))
#shuffle(train_names)
(len(train_names)-len(train_samples))/3
for i, train_path in enumerate(train_paths_mask_aug):
    image_names = glob.glob(train_path + "\*[!_GT].png")
    for image_name in image_names:
        #print(image_name)
        train_names.append((image_name, True))
print(len(train_names))
# Store data (serialize)
with open(dst_path, 'wb') as handle:
    pickle.dump((train_names,test_samples), handle, protocol=pickle.HIGHEST_PROTOCOL)
