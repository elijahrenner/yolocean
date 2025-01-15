# src/evaluation.py

import os
import yaml
import argparse
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from utils import evaluate_map50, display_curves

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(model_path, config):
    model = YOLO(model_path)
    metrics, map50 = evaluate_map50(model, config['paths']['output'], dataset='test')
    print(f"The mAP of the model on the test dataset is {map50}")
    return model, metrics

def visualize_evaluation(model, config):
    test_path = os.path.join(config['paths']['output'], 'runs', 'segment', 'train')
    display_curves(test_path)

def main(config):
    model_path = config['model_save_path']
    model, metrics = evaluate_model(model_path, config)
    visualize_evaluation(model, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11 Segmentation Model")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
