"""
Package Name: data_preparation
Description:
    Initialization module for dataset preparation scripts. This package
    contains utilities that take the DeepPCB and Kaggle PCB datasets
    and convert them into the unified directory structure expected by the
    CNN and YOLOv8 architectures.

Modules:
    - prepare_deeppcb: Prepares the DeepPCB dataset by generating defect
      patches and YOLOv8-style labels and splits.
    - prepare_kaggle_pcb: Prepares the Kaggle PCB dataset in the same unified
      format, including the 70/15/15 train/val/test split.

Typical Usage:
    from data_prep.prepare_deeppcb import main as prepare_deeppcb
    from data_prep.prepare_kaggle_pcb import main as prepare_kaggle_pcb
"""
