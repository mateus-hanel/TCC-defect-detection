import numpy as np
import pickle
import glob
import cv2
from shutil import copyfile

dataset_path = r".\datasets\KSDD2\train"

# Save defects in one folder, good images on another
ok_folder =  r".\datasets\KSDD2\train_ok"
defects_folder =  r".\datasets\KSDD2\train_defects"


test_names = glob.glob(dataset_path + "\*_GT.png")

defect_list = []
for i, name in enumerate(test_names):
    image = cv2.imread(name,0)
    if cv2.countNonZero(image) != 0:
        print(i)
        name_final = name.split("\\")[-1].split('.')[0][:-3]
        defect_list.append(name_final)
        dst = defects_folder + "\\" + name_final + "_GT.png"
        copyfile(name, dst)
        real = name[:-7] + ".png"
        dst = defects_folder + "\\" + name_final + ".png"
        copyfile(real,dst)
    else:
        name_final = name.split("\\")[-1].split('.')[0][:-3]
        dst = ok_folder + "\\" + name_final + "_GT.png"
        copyfile(name, dst)
        real = name[:-7] + ".png"
        dst = ok_folder + "\\" + name_final + ".png"
        copyfile(real,dst)
    
# Store data (serialize)
'''
with open('defect_list.pyb', 'wb') as handle:
    pickle.dump(defect_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('defect_list.pyb', "rb") as f:
    foo= pickle.load(f)
'''
