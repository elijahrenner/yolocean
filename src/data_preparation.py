# src/data_preparation.py

import os
import shutil
import random
import argparse
import yaml
import cv2
from utils import write_polygon_file
from tqdm import tqdm
import logging
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_directories(op_path):
    op_labels_path = os.path.join(op_path, 'labels')
    op_images_path = os.path.join(op_path, 'images')

    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(op_images_path, subset), exist_ok=True)
        os.makedirs(os.path.join(op_labels_path, subset), exist_ok=True)
    return op_labels_path, op_images_path

def sort_files(train_val_path, test_path):
    train_images_path = os.path.join(train_val_path, 'images')
    train_masks_path = os.path.join(train_val_path, 'masks')
    test_images_path = os.path.join(test_path, 'images')
    test_masks_path = os.path.join(test_path, 'masks')

    train_imgs_list = sorted(os.listdir(train_images_path))
    train_masks_list = sorted(os.listdir(train_masks_path))
    test_imgs_list = sorted(os.listdir(test_images_path))
    test_masks_list = sorted(os.listdir(test_masks_path))
    test_masks_list = list(filter(lambda x: x.endswith('.bmp'), test_masks_list))

    return train_imgs_list, train_masks_list, test_imgs_list, test_masks_list

def create_color_mappings():
    color_class_mapping = {
        (0, 0, 0): 0,
        (0, 0, 255): 1,
        (0, 255, 0): 2,
        (0, 255, 255): 3,
        (255, 0, 0): 4,
        (255, 0, 255): 5,
        (255, 255, 0): 6,
        (255, 255, 255): 7
    }

    color_name_mapping = {
        (0, 0, 0): 'black',
        (0, 0, 255): 'blue',
        (0, 255, 0): 'green',
        (0, 255, 255): 'sky',
        (255, 0, 0): 'red',
        (255, 0, 255): 'pink',
        (255, 255, 0): 'yellow',
        (255, 255, 255): 'white'
    }

    return color_class_mapping, color_name_mapping

def process_masks(train_masks_path, train_val_images_path, train_imgs_list, train_masks_list, color_class_mapping, color_name_mapping, op_labels_trainpath, op_images_trainpath, op_path):
    total_images_train = len(train_imgs_list)
    logging.info(f"Total training images: {total_images_train}")

    m = 0  # Counter for labels created

    for img in tqdm(train_masks_list, desc="Processing training masks"):
        mask_path = os.path.join(train_masks_path, img)
        image = cv2.imread(mask_path)  # Try to load the mask

        if image is None:  # Check if mask loading failed
            logging.warning(f"Failed to load mask: {mask_path}. Skipping.")
            continue  # Skip to the next mask

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pixels = image.reshape((-1, 3))
        unique_colors = np.unique(pixels, axis=0)  # Getting unique colors present in mask

        # Getting only those unique colors which are defined in problem
        unique_colors_defined = [tuple(value) for value in unique_colors if tuple(value) in color_class_mapping]
        unique_colors_defined = list(unique_colors_defined)
        total_colors = len(unique_colors_defined)

        img_name = os.path.splitext(img)[0]  # Extracting mask name
        img_full_name = img_name + '.jpg'  # Assuming corresponding image is .jpg

        source_image_path = os.path.join(train_val_images_path, img_full_name)  # Source image path
        destination_image_path = os.path.join(op_images_trainpath, img_full_name)  # Destination image path

        # Check if source image exists
        if not os.path.exists(source_image_path):
            logging.warning(f"Source image '{source_image_path}' does not exist. Skipping.")
            continue  # Skip if image doesn't exist

        # Copy image to output train path
        shutil.copy(source_image_path, destination_image_path)
        logging.info(f"Copied image '{source_image_path}' to '{destination_image_path}'")

        H, W, _ = image.shape  # Extracting mask dimensions
        class_contour_mapping = {}

        for color in unique_colors_defined:  # Looping through all defined colors present in mask
            class_code = color_class_mapping[color]  # Getting color-code
            # Convert color to a tuple of integers
            color = tuple(int(c) for c in color)
            mask = cv2.inRange(image, color, color)  # Getting mask of color on the mask-image
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Finding contours
            class_contour_mapping[class_code] = contours  # Mapping color-code to contours

        # Writing label text file
        write_polygon_file(class_contour_mapping, H, W, op_labels_trainpath, img_name)
        m += 1

    logging.info(f"Total number of labels created: {m}")

def process_test_masks(test_masks_path, test_images_path, test_masks_list, color_class_mapping, color_name_mapping, op_labels_testpath, op_images_testpath):
    m = 0  # Counter for labels created

    for img in tqdm(test_masks_list, desc="Processing test masks"):
        mask_path = os.path.join(test_masks_path, img)
        image = cv2.imread(mask_path)  # Try to load the mask

        if image is None:  # Check if mask loading failed
            logging.warning(f"Failed to load mask: {mask_path}. Skipping.")
            continue  # Skip to the next mask

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pixels = image.reshape((-1, 3))
        unique_colors = np.unique(pixels, axis=0)
        unique_colors_defined = [tuple(value) for value in unique_colors if tuple(value) in color_class_mapping]
        unique_colors_defined = list(unique_colors_defined)
        total_colors = len(unique_colors_defined)
        img_name = os.path.splitext(img)[0]
        img_full_name = img_name + '.jpg'  # Assuming corresponding image is .jpg

        source_image_path = os.path.join(test_images_path, img_full_name)  # Source image path
        destination_image_path = os.path.join(op_images_testpath, img_full_name)  # Destination image path

        # Check if source image exists
        if not os.path.exists(source_image_path):
            logging.warning(f"Source image '{source_image_path}' does not exist. Skipping.")
            continue  # Skip if image doesn't exist

        # Copy image to output test path
        shutil.copy(source_image_path, destination_image_path)
        logging.info(f"Copied image '{source_image_path}' to '{destination_image_path}'")

        H, W, _ = image.shape
        class_contour_mapping = {}
        for color in unique_colors_defined:
            class_code = color_class_mapping[color]
            # **Ensure color is a tuple of integers**
            color = tuple(int(c) for c in color)
            mask = cv2.inRange(image, color, color)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            class_contour_mapping[class_code] = contours

        write_polygon_file(class_contour_mapping, H, W, op_labels_testpath, img_name)
        m += 1

    logging.info(f"Total number of labels created: {m}")

def create_validation_data(train_imgs_list, train_masks_list, validation_size=110):
    combined = list(zip(train_imgs_list, train_masks_list))
    random.shuffle(combined)
    validation_data = combined[:validation_size]
    training_data = combined[validation_size:]
    return training_data, validation_data

def move_validation_data(validation_data, op_images_trainpath, op_labels_path):
    for img, mask in tqdm(validation_data, desc="Moving validation data"):
        source_image = os.path.join(op_images_trainpath, img)
        dest_image = os.path.join(op_images_trainpath.replace('train', 'val'), img)
        shutil.move(source_image, dest_image)
        logging.info(f"Moved image from '{source_image}' to '{dest_image}'")
   
        img_name = os.path.splitext(img)[0]
        label_file = img_name + '.txt'
        source_label = os.path.join(op_labels_path, 'train', label_file)
        dest_label = os.path.join(op_labels_path, 'val', label_file)
        shutil.move(source_label, dest_label)
        logging.info(f"Moved label from '{source_label}' to '{dest_label}'")

def write_config_file(config, op_path):
    newline = '\n'
    ln_1 = '# Train/val/test sets' + newline  # Starting with a comment line

    # Get absolute paths
    train_path = os.path.abspath(os.path.join(op_path, 'images', 'train'))
    val_path = os.path.abspath(os.path.join(op_path, 'images', 'val'))
    test_path = os.path.abspath(os.path.join(op_path, 'images', 'test'))

    # Train, val and test path declaration with absolute paths
    ln_2 = 'train: ' + "'" + train_path + "'" + newline
    ln_3 = 'val: ' + "'" + val_path + "'" + newline
    ln_4 = 'test: ' + "'" + test_path + "'" + newline
    ln_5 = newline

    # Names of the classes declaration
    ln_6 = 'names:' + newline
    ln_7 = '  0: Background' + newline
    ln_8 = '  1: Human divers' + newline
    ln_9 = '  2: Aquatic plants and sea-grass' + newline
    ln_10 = '  3: Wrecks and ruins' + newline
    ln_11 = '  4: Robots' + newline
    ln_12 = '  5: Reefs and invertebrates' + newline
    ln_13 = '  6: Fish and vertebrates' + newline
    ln_14 = '  7: Sea-floor and rocks'

    config_lines = [ln_1, ln_2, ln_3, ln_4, ln_5, ln_6, ln_7, ln_8, ln_9, ln_10, ln_11, ln_12, ln_13, ln_14]

    # Creating path for config file
    config_path = os.path.join(op_path, 'config.yaml')

    # Writing config file
    with open(config_path, 'w') as f:
        f.writelines(config_lines)

    print(f"Configuration file written to {config_path}")

def initialize_logging(op_path):
    log_dir = os.path.join(op_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, 'data_preparation.log'),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.info("Data Preparation Logging initialized.")

def main(config):
    initialize_logging(config['paths']['output'])
    ip_path = config['paths']['input']
    op_path = config['paths']['output']

    train_val_path = os.path.join(ip_path, 'train_val')
    test_path = os.path.join(ip_path, 'TEST')

    op_labels_path, op_images_path = prepare_directories(op_path)

    train_imgs_list, train_masks_list, test_imgs_list, test_masks_list = sort_files(train_val_path, test_path)

    color_class_mapping, color_name_mapping = create_color_mappings()

    # Define the path to source training images
    train_val_images_path = os.path.join(train_val_path, 'images')

    # Process training masks
    process_masks(
        train_masks_path=os.path.join(train_val_path, 'masks'),
        train_val_images_path=train_val_images_path,
        train_imgs_list=train_imgs_list,
        train_masks_list=train_masks_list,
        color_class_mapping=color_class_mapping,
        color_name_mapping=color_name_mapping,
        op_labels_trainpath=os.path.join(op_labels_path, 'train'),
        op_images_trainpath=os.path.join(op_images_path, 'train'),
        op_path=op_path
    )

    # Define the path to source test images
    test_images_path = os.path.join(test_path, 'images')

    # Process test masks
    process_test_masks(
        test_masks_path=os.path.join(test_path, 'masks'),
        test_images_path=test_images_path,
        test_masks_list=test_masks_list,
        color_class_mapping=color_class_mapping,
        color_name_mapping=color_name_mapping,
        op_labels_testpath=os.path.join(op_labels_path, 'test'),
        op_images_testpath=os.path.join(op_images_path, 'test')
    )

    # Create validation dataset
    logging.info("Creating validation dataset...")
    training_data, validation_data = create_validation_data(train_imgs_list, train_masks_list, validation_size=110)
    move_validation_data(validation_data, os.path.join(op_images_path, 'train'), op_labels_path)

    # Write config file for YOLO
    write_config_file(config, op_path)

    logging.info("Data preparation completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preparation for YOLOv8 Segmentation")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
