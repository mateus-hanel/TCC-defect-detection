import numpy as np
import pickle5 as pickle
import glob
from random import shuffle
import os
fn = f"STEEL/split_300_150_flip.pyb"
with open(f"splits/{fn}", "rb") as f:
    train_images, test_images, validation_images = pickle.load(f)
validation_image
train_images[1]
train_images
# Validation é no
# Tentar editar apenas o train e o validation
dst_path = ".\splits\STEEL\split_300_10_new.pyb"

train_path = r".\datasets\STEEL\train_images"

new_train = []
for train in train_images:
    l = list(train)
    l[0] = os.path.join(train_path,l[0]+".jpg")
    new_train.append(tuple(l))

new_val = []
for val in validation_images:
    l = list(val)
    l[0] = os.path.join(train_path,l[0]+".jpg")
    new_val.append(tuple(l))

with open(dst_path, 'wb') as handle:
    pickle.dump((new_train,test_images,new_val), handle, protocol=pickle.HIGHEST_PROTOCOL)
