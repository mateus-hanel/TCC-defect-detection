import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image

def rle_to_mask(rle, image_size):
    if len(rle) % 2 != 0:
        raise Exception('Suspicious')

    w, h = image_size
    mask_label = np.zeros(w * h, dtype=np.float32)

    positions = rle[0::2]
    length = rle[1::2]
    for pos, le in zip(positions, length):
        mask_label[pos - 1:pos + le - 1] = 1
    mask = np.reshape(mask_label, (h, w), order='F')
    return mask

def read_annotations(fn):
    arr = np.array(pd.read_csv(fn), dtype=np.object)
    annotations_dict = {}
    for sample, _, rle in arr:
        #img_name = sample[:-4]
        img_name = sample

        annotations_dict[img_name] = rle

    return annotations_dict

annotations = read_annotations(r".\datasets\STEEL\train.csv")
len(annotations)
image_size = (1600, 256)
dst = r".\datasets\STEEL\masks"

for key, value in annotations.items():
    print(key)
    rle = list(map(int,value.split(" ")))
    mask = (rle_to_mask(rle, image_size)*255).astype(np.uint8)
    im = Image.fromarray(mask)

    im.save(key[:-4] + "_GT.png")
