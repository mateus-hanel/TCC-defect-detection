import numpy as np
import pickle
import glob
import cv2

dataset_path = r".\datasets\KSDD2\train"

test_names = glob.glob(dataset_path + "\*_GT.png")
i = 0
defect_list = []
for name in test_names:
    image = cv2.imread(name,0)
    if cv2.countNonZero(image) != 0:
        print(i)
        i+=1
        name = name.split("\\")[-1].split('.')[0][:-3]
        print(name)
        defect_list.append(name)

print(len(defect_list))

# Store data (serialize)
with open('defect_list.pyb', 'wb') as handle:
    pickle.dump(defect_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('defect_list.pyb', "rb") as f:
    foo= pickle.load(f)
foo
train_samples
