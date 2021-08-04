# Flips all images on the dataset 180 degrees and saves to a folder

import cv2
import glob
from matplotlib import pyplot as plt
import numpy as np
from glob import glob
import os
paths = glob('*/')
paths
for path in paths:
    files = glob(path + "*.bmp")
    for file in files:
        img = cv2.imread(file)
        n_white_pix = np.sum(img == 255)
        if n_white_pix != 0:
            original = file[:-10] + ".jpg"
            dst_file = file[:5] + "_defect" + file[5:]
            dst_original = original[:5] + "_defect" + original[5:]

            #os.remove(original)
            #os.remove(file)
            print(original)
            os.rename(original, dst_original)
            os.rename(file, dst_file)
