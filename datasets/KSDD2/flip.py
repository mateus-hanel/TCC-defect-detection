# Flips all images on the dataset 180 degrees and saves to a folder
'''
import cv2
import glob
from matplotlib import pyplot as plt

src = r"./train/"
dst_180 = r"./train_180/"

dst_v = r"./train_flipped_vertical/"
dst_h = r"./train_flipped_horizontal/"


dst_v = r"./train_flipped_vertical/"
src_images =  glob.glob(src + "*.png")

for img_path in src_images:
    img_name = img_path.split("\\")[-1]
    print(img_name)

    img = cv2.imread(img_path)

    img_rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
    img_fliped_vertical = cv2.flip(img, 0)
    img_fliped_horizontal = cv2.flip(img, 1)

    cv2.imwrite(dst_180 + img_name, img_rotated_180)
    cv2.imwrite(dst_v + img_name, img_fliped_vertical)
    cv2.imwrite(dst_h + img_name, img_fliped_horizontal)
    #break


plt.imshow(img)
plt.imshow(img_rotated_180)
plt.imshow(img_fliped_vertical)
plt.imshow(img_fliped_horizontal)
'''
part = 120
print(f"{part}.png")
