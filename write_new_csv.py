import csv
import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np

def rle_encoding(x, normalized):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    if normalized:
        dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    else:
        dots = np.where(x.T.flatten()==255)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


src = r'.\datasets\STEEL\split_300_all'

dataset_paths = [r'.\datasets\STEEL\split_300_all',
r'.\datasets\STEEL\split_300_all_180',
r'.\datasets\STEEL\split_300_all_horizontal',
r'.\datasets\STEEL\split_300_all_vertical'
]

paths = glob.glob(src + r"\*.png")
error = 0
with open(r'.\datasets\STEEL\train.csv', 'a',newline='') as f:
    writer = csv.writer(f)
    for path in paths:
        name = path.split("\\")[-1][:-7] + ".jpg"
        #print(name)

        for dataset_path in dataset_paths:
            try:
                final_name = dataset_path + "\\" + name

                GT = cv2.imread(final_name[:-4] + "_GT.png",0)
                rle = rle_encoding(GT,False)
                rle_str = []
                for element in rle:
                    rle_str.append(str(element))
                rle_str = " ".join(rle_str)
                fields=[final_name,'1',rle_str]
                writer.writerow(fields)
            except:
                error+=1
                print("error counter: " + str(error))
                print(final_name)
                quit()
print("FINISHED")
        #break
