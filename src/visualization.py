# src/visualization.py

import os
import matplotlib.pyplot as plt
import cv2
import random
import copy
import numpy as np

def plot_image_and_mask(train_images_path, train_masks_path, train_imgs_list, train_masks_list):
    figure, axis = plt.subplots(1, 2, figsize=(30, 30))
    plt.axis('off')
    k = random.randint(0, len(train_imgs_list) - 1)  # choosing any random image number

    img_path = os.path.join(train_images_path, train_imgs_list[k])  # defining image path
    mask_path = os.path.join(train_masks_path, train_masks_list[k])  # defining mask path

    img_title = os.path.basename(img_path)  # extracting image filename from path
    mask_title = os.path.basename(mask_path)  # extracting mask filename from path

    # displaying image and mask
    axis[0].imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    axis[0].set_title(img_title, fontsize=30)
    axis[0].set_xticks([])
    axis[0].set_yticks([])
    axis[1].imshow(cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB))
    axis[1].set_title(mask_title, fontsize=30)
    axis[1].set_xticks([])
    axis[1].set_yticks([])

    plt.tight_layout()
    plt.show()

def draw_contour_for_one_color_on_mask(train_images_path, train_masks_path, train_imgs_list, train_masks_list, color_class_mapping, color_name_mapping):
    figure, axis = plt.subplots(1, 3, figsize=(30, 30))
    plt.axis('off')
    k = random.randint(0, len(train_imgs_list) - 1)  # choosing any random image

    img_path = os.path.join(train_images_path, train_imgs_list[k])  # defining image path
    img_title = os.path.basename(img_path)  # extracting basename from path
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    mask_path = os.path.join(train_masks_path, train_masks_list[k])  # defining mask path
    mask_title = os.path.basename(mask_path)  # extracting basename from path
    mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)

    mask_copy = copy.deepcopy(mask)  # creating a copy of mask to draw contour
    pixels = mask.reshape((-1, 3))
    unique_colors = np.unique(pixels, axis=0)  # getting unique colors present in mask

    # getting only those unique colors which are defined in problem
    unique_colors_defined = [tuple(color) for color in unique_colors if tuple(color) in color_class_mapping]
    total_colors = len(unique_colors_defined)
    if total_colors == 0:
        print("No defined colors found in the mask.")
        return
    j = random.randint(0, total_colors - 1)  # selecting any random color among all the defined colors present in mask
    color = unique_colors_defined[j]
    # defining title for contour on mask
    mask_copy_title = os.path.basename(mask_path) + ' with contour on color ' + str(color_name_mapping[color])

    mask_mask = cv2.inRange(mask, color, color)  # getting mask of the selected color on the mask-image
    contours, _ = cv2.findContours(mask_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # finding contours
    orange = (255, 85, 0)
    cv2.drawContours(mask_copy, contours, -1, orange, 4)  # drawing contours

    axis[0].imshow(img)  # displaying image
    axis[0].set_title(img_title, fontsize=30)
    axis[0].set_xticks([])
    axis[0].set_yticks([])
    axis[1].imshow(mask)  # displaying mask
    axis[1].set_title(mask_title, fontsize=30)
    axis[1].set_xticks([])
    axis[1].set_yticks([])
    axis[2].imshow(mask_copy)  # displaying copy of mask with contour
    axis[2].set_title(mask_copy_title, fontsize=30)
    axis[2].set_xticks([])
    axis[2].set_yticks([])

    plt.tight_layout()
    plt.show()
