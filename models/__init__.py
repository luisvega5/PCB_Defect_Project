"""
Module Name: models
Description:
    Package for all neural network architectures used in the PCB Defect
    Detection project. This module currently exposes the DefectCNN model,
    which is a convolutional neural network designed for image-based PCB
    defect classification. Additional model architectures can be added to
    this package in the future and re-exported here for convenient import
    from the rest of the codebase.
Contents:
    DefectCNN:
        Convolutional neural network defined in cnn_model.py. It takes RGB
        PCB images or cropped defect patches as input and produces class
        logits for the configured number of defect categories.
Usage:
    from models import DefectCNN
"""