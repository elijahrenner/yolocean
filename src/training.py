# src/training.py

import os
import yaml
import argparse
from ultralytics import YOLO
import wandb
import shutil
import cv2
import matplotlib.pyplot as plt
from utils import write_polygon_file
from tqdm import tqdm
import logging

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def initialize_logging(op_path):
    log_dir = os.path.join(op_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, 'training.log'),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.info("Logging initialized.")

def initialize_wandb(wandb_config):
    if wandb_config.get('enable', False):
        api_key = os.getenv('WANDB_API_KEY', wandb_config.get('api_key', ''))
        if api_key:
            wandb.login(key=api_key)
            wandb.init(project="yolocean")  # Replace with your project name
            logging.info("W&B initialized.")
            return True
        else:
            logging.warning("W&B is enabled but no API key provided. Skipping W&B initialization.")
    return False

def prepare_output_directories(op_path):
    segment_path = os.path.join(op_path, 'segment', 'train')
    os.makedirs(segment_path, exist_ok=True)
    return segment_path

def train_model(config, segment_path, wandb_enabled):
    model_config = config['training']['model_config']
    pretrained_weights = config['training']['pretrained_weights']
    epochs = config['training']['epochs']
    degrees = config['training']['augmentation']['degrees']
    shear = config['training']['augmentation']['shear']
    perspective = config['training']['augmentation']['perspective']

    model = YOLO(model_config).load(pretrained_weights)
    logging.info(f"Starting training with config: {model_config}, pretrained weights: {pretrained_weights}")
    
    results = model.train(
        data=os.path.join(config['paths']['output'], 'config.yaml'),
        epochs=epochs,
        degrees=degrees,
        shear=shear,
        perspective=perspective,
        show_boxes=False  # Updated parameter
    )
    
    if wandb_enabled:
        wandb.finish()
        logging.info("W&B training session finished.")
        
    return model, results

def plot_training_results(segment_path):
    trainingresult_path = os.path.join(segment_path, 'train')
    results_png = cv2.imread(os.path.join(trainingresult_path, 'results.png'))
    if results_png is not None:
        plt.figure(figsize=(30, 30))
        plt.imshow(cv2.cvtColor(results_png, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Training Results")
        plt.show()
        logging.info("Training results plotted.")
    else:
        logging.error(f"Results image not found at {os.path.join(trainingresult_path, 'results.png')}")

def save_model(model, save_path):
    model.save(save_path)
    logging.info(f"Model saved at {save_path}")

def main(config):
    initialize_logging(config['paths']['output'])
    wandb_enabled = initialize_wandb(config.get('wandb', {}))
    segment_path = prepare_output_directories(config['paths']['output'])
    model, results = train_model(config, segment_path, wandb_enabled)
    plot_training_results(segment_path)
    save_model(model, config['model_save_path'])
    logging.info("Training completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Segmentation Model")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
