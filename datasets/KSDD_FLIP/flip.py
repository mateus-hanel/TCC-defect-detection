# Flips all images on the dataset 180 degrees and saves to a folder

import cv2
import glob
from matplotlib import pyplot as plt

from glob import glob
paths = glob('*/')
paths
for path in paths:
    files = glob(path + "*")
    print(files)
    for file in files:
        img = cv2.imread(file)

        img_rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
        img_fliped_vertical = cv2.flip(img, 0)
        img_fliped_horizontal = cv2.flip(img, 1)
        if file[-3:] == "jpg":
            name_180 = file[:-4] + "_180" + file[-4:]
            name_v = file[:-4] + "_v" + file[-4:]
            name_h = file[:-4] + "_h" + file[-4:]
        else:
            files[1][:-10] + "_180" + files[1][-10:]
            name_180 = file[:-10] + "_180" + file[-10:]
            name_v = file[:-10] + "_v" + file[-10:]
            name_h = file[:-10] + "_h" + file[-10:]
        cv2.imwrite(name_180, img_rotated_180)
        cv2.imwrite(name_v, img_fliped_vertical)
        cv2.imwrite(name_h, img_fliped_horizontal)
