"""
Package Name: train_utils
Description:
    Initialization module for training and evaluation pipelines used in the
    PCB defect detection project. This package wraps the CNN and YOLOv8
    training logic into reusable functions that can be called from main.py
    or from external scripts.

Modules:
    - train_cnn: Contains the train_and_evaluate_cnn function that trains the
      CNN classifier and reports accuracy metrics.
    - train_yolov8: Contains helper functions that fine-tune YOLOv8 on the
      prepared datasets and evaluate mAP on the test split.

Typical Usage:
    from training.train_cnn import train_and_evaluate_cnn
    from training.train_yolov8 import train_yolov8
"""
