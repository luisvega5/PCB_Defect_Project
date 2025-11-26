"""
Package Name: model_configuration
Description:
    Initialization module for dataset utilities used in the PCB defect
    detection project. This package provides PyTorch dataset classes and
    configuration dictionaries that describe each supported dataset.

Modules:
    - dataset_cnn: Implements the PCBCNNDataset and data loader helpers
      used by the CNN training.
    - dataset_configs: Contains dataset metadata such as root paths,
      class names, and YOLOv8 data.yaml locations for DeepPCB and Kaggle PCB.

Typical Usage:
    from datasets.dataset_cnn import PCBCNNDataset
    from datasets.dataset_configs import DATASET_CONFIGS
"""
