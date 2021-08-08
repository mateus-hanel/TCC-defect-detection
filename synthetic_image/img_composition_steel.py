import os
import cv2
import random
import math
import time
import numpy as np
from scipy.stats import truncnorm
from matplotlib import pyplot as plt
from glob import glob
import imgaug.augmenters as iaa
from transforms import ElasticDistortion
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imutils


class DefectGenerator:
    def __init__(self, dir):
        self.dir = dir
        defect_list = os.listdir(dir)
        img_list = [x for x in defect_list if "_GT" not in x]

        mask_list = [x for x in defect_list if "_GT" in x]

        area_list = []
        for name in img_list:
            mask_path = os.path.join(dir, name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            area = np.sum(mask > 0)
            area_list.append(area)

        area_order = np.argsort(area_list)[::-1]

        self.file_names = np.array(img_list)[area_order]
        np.random.shuffle(self.file_names)
        print(self.file_names)
        self.loops = 0
        self.defect_index = 0
        self.max_index = len(self.file_names) - 1

    def gen_defect(self, augmenter):
        file_name = self.file_names[self.defect_index]
        print(self.defect_index, self.max_index)
        print(file_name)
        if self.defect_index == self.max_index:
            self.defect_index = 0
            self.loops += 1
        else:
            self.defect_index += 1
        img_path = os.path.join(self.dir, file_name)
        mask_path = os.path.join(self.dir, file_name[:-4]+"_GT.png")
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        seg_map = SegmentationMapsOnImage(mask, mask.shape)
        aug_img, aug_mask = augmenter(images=[img], segmentation_maps=[seg_map])
        aug_img = aug_img[0]
        aug_mask = aug_mask[0].get_arr()

        return aug_img, aug_mask

def add_offset(img, type=cv2.BORDER_CONSTANT, value=None):
    size = int(np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2))
    of_h = int((size - img.shape[0]) / 2)
    of_w = int((size - img.shape[1]) / 2)

    if value is None:
        ch = 1 if len(img.shape) == 2 else img.shape[-1]
        value = tuple([0] * ch)

    return cv2.copyMakeBorder(img, of_h, of_h, of_w, of_w, type, value=value)

def remove_offset(img, shape):
    of_h = int((img.shape[0] - shape[0]) / 2)
    of_w = int((img.shape[1] - shape[1]) / 2)
    return img[of_h:of_h+shape[0], of_w:of_w+shape[1]]

def img_offset(images, random_state, parents, hooks):
    return np.array([add_offset(img) for img in images])

def seg_map_offset(seg_maps, random_state, parents, hooks):
    res = []
    for seg_map in seg_maps:
        mask = add_offset(seg_map.get_arr())
        res.append(SegmentationMapsOnImage(mask, mask.shape))
    return res

def blob_brect(mask):
    br, bc = np.where(mask > 0)
    b_min_r, b_max_r = np.min(br), np.max(br)
    b_min_c, b_max_c = np.min(bc), np.max(bc)

    return b_min_c, b_min_r, b_max_c + 1, b_max_r + 1

def try_compose(src_img, src_mask, dst_img, dst_mask, pt=None):

    comp_mask = np.uint8(src_mask > 0) * 255

    comp_mask = cv2.dilate(comp_mask, np.ones((3, 3)))

    # Obtém a bounding box da marcação do defeito
    min_x, min_y, max_x, max_y = blob_brect(comp_mask)

    src_h = max_y-min_y
    src_w = max_x-min_x

    dst_h, dst_w = dst_img.shape[:2]

    src_img = src_img[min_y:max_y, min_x:max_x]
    src_mask = src_mask[min_y:max_y, min_x:max_x]
    comp_mask = comp_mask[min_y:max_y, min_x:max_x]

    x0, y0 = round(src_w / 2), round(src_h / 2)
    x1, y1 = dst_w - x0 - 1, dst_h - y0 - 1

    if x0 > x1:
        src_img = imutils.resize(src_img, width=dst_w-2,inter = cv2.INTER_NEAREST)
        src_mask = imutils.resize(src_mask, width=dst_w-2,inter = cv2.INTER_NEAREST)
        comp_mask = imutils.resize(comp_mask, width=dst_w-2,inter = cv2.INTER_NEAREST)
        src_h, src_w = src_img.shape[:2]

        x0, y0 = round(src_w / 2), round(src_h / 2)
        x1, y1 = dst_w - x0 - 1, dst_h - y0 - 1

    if y0>y1:
        src_img = imutils.resize(src_img, height=dst_h-2,inter = cv2.INTER_NEAREST)
        src_mask = imutils.resize(src_mask, height=dst_h-2,inter = cv2.INTER_NEAREST)
        comp_mask = imutils.resize(comp_mask, height=dst_h-2,inter = cv2.INTER_NEAREST)
        src_h, src_w = src_img.shape[:2]

        x0, y0 = round(src_w / 2), round(src_h / 2)
        x1, y1 = dst_w - x0 - 1, dst_h - y0 - 1

    n_white_pix = 1
    counter = 10
    while n_white_pix != 0 and counter != 0:
        if pt is None:
            cx = random.randint(x0, x1)
            cy = random.randint(y0, y1)
        else:
            cx, cy = pt

        dst_region = dst_mask[int(cy-y0):int(cy-y0+src_h), int(cx-x0):int(cx-x0+src_w)]
        n_white_pix = np.sum(dst_region == 255)
        counter -= 1
        print("counter:" + str(counter))
    res_mask = dst_mask.copy()
    res_mask[int(cy-y0):int(cy-y0+src_h), int(cx-x0):int(cx-x0+src_w)] += src_mask
    #res_img = cv2.seamlessClone(src_img, dst_img, comp_mask, (cx, cy), cv2.MONOCHROME_TRANSFER)
    #res_img = cv2.seamlessClone(src_img, dst_img, comp_mask, (cx, cy), cv2.MIXED_CLONE)
    res_img = cv2.seamlessClone(src_img, dst_img, comp_mask, (cx, cy), cv2.NORMAL_CLONE)


    return res_img, res_mask, True

bg_glob = r"..\datasets\STEEL\split_300_bg\*[!_GT].png"

fg_dir = r'.\steel_300_50_2'
out_dir = r'..\datasets\STEEL\split_300_synthetic_50'

gen_count = 1500 # numero de imagens na saída
max_loops = 4
defect_aug = iaa.Sequential([
    iaa.Lambda(func_images=img_offset,
               func_segmentation_maps=seg_map_offset),
    ElasticDistortion(150, 150, 3),
    iaa.PiecewiseAffine(scale=(0.01, 0.05)),
    iaa.Affine(rotate=(0, 180), shear=(-5, 5)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    #iaa.Multiply(1.3),
    iaa.Resize((0.9, 1.1))])


bg_paths = glob(bg_glob)

defect_gen = DefectGenerator(fg_dir)

#for i in range(gen_count):
i = 0
while defect_gen.loops < max_loops:
    # Finds a random background
    bg_img = cv2.imread(random.choice(bg_paths))
    comp_img = bg_img
    comp_mask = np.zeros(tuple(comp_img.shape[:2]), dtype=np.uint8)

    for j in range(3):
        defect_img, defect_mask = defect_gen.gen_defect(defect_aug)

        comp_img, comp_mask, success = try_compose(defect_img, defect_mask,
                                                           comp_img, comp_mask)
        comp_img = remove_offset(comp_img, bg_img.shape)
        comp_mask = remove_offset(comp_mask, bg_img.shape)


    '''
    defect_img, defect_mask = defect_gen.gen_defect(defect_aug)

    comp_img, comp_mask, success = try_compose(defect_img, defect_mask,
                                               comp_img, comp_mask)
    defect_img, defect_mask = defect_gen.gen_defect(defect_aug)

    comp_img, comp_mask, success = try_compose(defect_img, defect_mask,
                                               comp_img, comp_mask)

    comp_img, comp_mask, success = try_compose(defect_img, defect_mask,
                                               comp_img, comp_mask)
    '''
    #defect_gen.reset(scattering=random.uniform(0.001, 0.1))

    out_name = 'gen_{0}.png'.format(i)
    cv2.imwrite(os.path.join(out_dir, out_name), comp_img)
    cv2.imwrite(os.path.join(out_dir, out_name[:-4]+"_GT.png"), comp_mask)
    i += 1
