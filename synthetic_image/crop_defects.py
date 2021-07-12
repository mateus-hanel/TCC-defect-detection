import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

img_dir = r'..\datasets\KSDD2\train_defects'
defect_out_dir = r'cropped_defects_train'
mask_ext = '.png'
img_ext = '.png'
margin = 0

mask_out_dir = defect_out_dir

if not os.path.exists(mask_out_dir):
    os.makedirs(mask_out_dir)

img_list = os.listdir(img_dir)

mask_list = [os.path.basename(x).split('.')[0] for x in glob.glob(img_dir + "\*_GT.png")]
img_list= glob.glob(img_dir + "\*[!_GT].png")
mask_list = sorted(mask_list)
img_list = sorted(img_list)

for name in img_list:

    only_name = name.split("\\")[-1][:-4]
    print(only_name)
    img_path = name
    mask_path = name[:-4] + "_GT.png"

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    img_h, img_w = mask.shape


    cnt_list, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)[-2:]

    for defect_id, cnt in enumerate(cnt_list):
        a = cv2.contourArea(cnt)

        x, y, w, h = cv2.boundingRect(cnt)

        if w < 10 and h < 10:
            continue

        m_x0, m_x1 = min(margin, x), min(margin, img_w - x - w)
        m_y0, m_y1 = min(margin, y), min(margin, img_h - y - h)

        #if 0 in [m_x0, m_x1, m_y0, m_y1] and a < 10000:
        #    continue

        cnt[:, 0, 0] = cnt[:, 0, 0] - x + m_x0
        cnt[:, 0, 1] = cnt[:, 0, 1] - y + m_y0

        defect_mask_shape = (h + m_y0 + m_y1, w + m_x0 + m_x1)

        ext_defect_mask = np.zeros(defect_mask_shape, dtype=np.uint8)
        _ = cv2.drawContours(ext_defect_mask, [cnt], -1, 1, cv2.FILLED)

        c_r0, c_r1 = (y - m_y0), (y + h + m_y1)
        c_c0, c_c1 = (x - m_x0), (x + w + m_x1)

        defect_mask_crop = mask[c_r0:c_r1, c_c0:c_c1]
        defect_mask = defect_mask_crop * ext_defect_mask

        defect_img = img[c_r0:c_r1, c_c0:c_c1]
        plt.imshow(defect_img)
        name_no_ext = os.path.splitext(name)[0]
        defect_name = '{0}_{1}.png'.format(only_name, defect_id)
        mask_name = defect_name[:-4] + "_GT.png"

        cv2.imwrite(os.path.join(defect_out_dir, defect_name), defect_img)
        cv2.imwrite(os.path.join(defect_out_dir, mask_name), defect_mask)
