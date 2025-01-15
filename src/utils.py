# src/utils.py

import os
import cv2
import copy
import numpy as np
import yaml
import argparse
from collections import defaultdict
from matplotlib import pyplot as plt

def write_polygon_file(class_contour_mapping, H, W, output_path, img_name):
    coordinates = {}
    for obj in class_contour_mapping:  # looping through all classes present in the mask
        polygons = []
        for cnt in class_contour_mapping[obj]:  # looping through all contours present in the class
            if cv2.contourArea(cnt) > 20:  # neglecting very small contours
                polygon = []
                for point in cnt:  # looping through all points present in the contour
                    x, y = point[0]
                    polygon.append(round(x / W, 4))
                    polygon.append(round(y / H, 4))
                polygons.append(polygon)
        coordinates[obj] = polygons

    # creating text file for all contours of all classes present in mask
    with open('{}.txt'.format(os.path.join(output_path, img_name)), 'w') as f:
        for obj in coordinates:
            for polygon in coordinates[obj]:
                for p_, p in enumerate(polygon):
                    if p_ == len(polygon) - 1:  # if point is the last point in contour, need to give newline
                        f.write('{}\n'.format(p))
                    elif p_ == 0:  # if point is the first point in contour, need to specify class also
                        f.write('{} {} '.format(obj, p))
                    else:  # any other point between first and last
                        f.write('{} '.format(p))

def evaluate_map50(trained_model, data_path, dataset='test'):
    metrics = trained_model.val(data=data_path, split=dataset)
    map50 = round(metrics.seg.map50, 3)
    print(f"The mAP of the model for all images on the {dataset} dataset is {map50}")
    return metrics, map50

def display_curves(root_path):
    plt.figure(figsize=(50, 50))

    # Displaying mask p curve
    p_curve_path = os.path.join(root_path, 'MaskP_curve.png')
    if os.path.exists(p_curve_path):
        p_curve = cv2.imread(p_curve_path)
        if p_curve is not None:
            ax = plt.subplot(5, 1, 1)
            plt.imshow(cv2.cvtColor(p_curve, cv2.COLOR_BGR2RGB))
            plt.title("Mask Precision Curve")
        else:
            print(f"Error loading image: {p_curve_path}")
    else:
        print(f"File not found: {p_curve_path}")

    # Displaying mask r curve
    r_curve_path = os.path.join(root_path, 'MaskR_curve.png')
    if os.path.exists(r_curve_path):
        r_curve = cv2.imread(r_curve_path)
        if r_curve is not None:
            ax = plt.subplot(5, 1, 2)
            plt.imshow(cv2.cvtColor(r_curve, cv2.COLOR_BGR2RGB))
            plt.title("Mask Recall Curve")
        else:
            print(f"Error loading image: {r_curve_path}")
    else:
        print(f"File not found: {r_curve_path}")

    # Displaying mask pr curve
    pr_curve_path = os.path.join(root_path, 'MaskPR_curve.png')
    if os.path.exists(pr_curve_path):
        pr_curve = cv2.imread(pr_curve_path)
        if pr_curve is not None:
            ax = plt.subplot(5, 1, 3)
            plt.imshow(cv2.cvtColor(pr_curve, cv2.COLOR_BGR2RGB))
            plt.title("Mask PR Curve")
        else:
            print(f"Error loading image: {pr_curve_path}")
    else:
        print(f"File not found: {pr_curve_path}")

    # Displaying mask f1 curve
    f1_curve_path = os.path.join(root_path, 'MaskF1_curve.png')
    if os.path.exists(f1_curve_path):
        f1_curve = cv2.imread(f1_curve_path)
        if f1_curve is not None:
            ax = plt.subplot(5, 1, 4)
            plt.imshow(cv2.cvtColor(f1_curve, cv2.COLOR_BGR2RGB))
            plt.title("Mask F1 Curve")
        else:
            print(f"Error loading image: {f1_curve_path}")
    else:
        print(f"File not found: {f1_curve_path}")

    # Displaying confusion matrix
    confusion_matrix_path = os.path.join(root_path, 'confusion_matrix.png')
    if os.path.exists(confusion_matrix_path):
        confusion_matrix = cv2.imread(confusion_matrix_path)
        if confusion_matrix is not None:
            ax = plt.subplot(5, 1, 5)
            plt.imshow(cv2.cvtColor(confusion_matrix, cv2.COLOR_BGR2RGB))
            plt.title("Confusion Matrix")
        else:
            print(f"Error loading image: {confusion_matrix_path}")
    else:
        print(f"File not found: {confusion_matrix_path}")

    plt.tight_layout()
    plt.show()
