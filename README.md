# yolocean

![Project Banner](images/example.gif)

## Table of Contents
1. [Overview](#overview)
2. [Dataset Preparation](#dataset-preparation)
3. [Installation](#installation)
4. [Training the Model](#training-the-model)
5. [Running Inference](#running-inference)
6. [File Structure](#file-structure)
7. [Credits](#credits)

## Overview
YOLOcean is a deep learning project for underwater image segmentation using the YOLO framework. It utilizes the SUIM dataset to train and test a YOLO-based segmentation model.

## Dataset Preparation
1. Access the SUIM dataset from [here](https://irvlab.cs.umn.edu/resources/suim-dataset).
2. Download and place the dataset into the `/data` folder.
3. Unzip `TEST.zip` and `train_val.zip` archives.
4. Prepare the data using the script:
   ```bash
   sh scripts/prepare_data.sh
   ```

## Installation
1. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Edit the settings file:
   ```bash
   nano /root/.config/Ultralytics/settings.json
   ```
   Set the datasets directory to the current directory:
   ```json
   {
     "datasets_dir": "."
   }
   ```

## Training the Model
Run the training script:
```bash
sh scripts/train_model.sh
```

## Running Inference
Run inference using the test script:
```bash
python tests/test.py
```

## File Structure
```
├── configs/
│   └── config.yaml
├── data/
│   └── SUIM/
│       ├── TEST/
│       │   ├── images/
│       │   └── masks/
│       ├── train_val/
│       │   ├── images/
│       │   └── masks/
│       └── INFO.txt
├── images/
│   └── example.gif
├── notebooks/
│   └── analysis.ipynb
├── outputs/
│   ├── images/
│   ├── labels/
│   ├── logs/
│   ├── models/
│   └── config.yaml
├── runs/
│   └── segment/
│       └── trainXX/
│           ├── weights/
│           ├── metrics and visualizations
├── scripts/
│   ├── evaluate_model.sh
│   ├── prepare_data.sh
│   └── train_model.sh
├── src/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── training.py
│   ├── utils.py
│   └── visualization.py
├── tests/
│   ├── test.py
│   └── test_utils.py
├── README.md
├── YOLO.ipynb
├── requirements.txt
├── settings.json
└── model files
```

## Credits
- SUIM Dataset: [SUIM Dataset](https://irvlab.cs.umn.edu/resources/suim-dataset)
- YOLO Framework: [Ultralytics](https://github.com/ultralytics)
- Contributors: [Elijah Renner](https://elijahrenner.com), Isabel Beckman, [Dr. Alberto Quattrini Li](https://web.cs.dartmouth.edu/people/alberto-quattrini-li)

---
For any issues or contributions, feel free to open an issue or submit a pull request. 
