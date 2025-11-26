"""
    ======================================================================
                            Dataset Configs Script
    ======================================================================
    Name:          dataset_configs.py
    Author:        Luis Obed Vega Maisonet
    University:    University of Central Florida (UCF)
    Course:        CAP 5415 - Computer Vision
    Academic Year: 2025
    Semester:      Fall
    Professor:     Dr. Yogesh Singh Rawat

    Description:
    This script centralizes all dataset-specific configuration in one place.
    It defines a dictionary of configurations keyed by dataset name, for
    example "deeppcb" and "kaggle_pcb". Each entry contains a description
    of the dataset, the number of classes, the ordered list of class names
    (which defines the mapping to class IDs), the path to the YOLO
    data.yaml file for that dataset, and the root directory for the CNN
    dataset directory tree. When main.py runs, it reads from this
    configuration to know how many classes to feed into the CNN, where to
    find the YOLO config, where the CNN data lives, and what to print as a
    description to the console. This makes it easy to switch datasets with
    a flag instead of hard-coding paths and class mappings.
------------------------------------------------Imports and Globals-----------------------------------------------------
"""
DATASET_CONFIGS = {
    "deeppcb": {
        "description": (
            "DeepPCB: 1,500 image pairs with 6 defect types "
            "(open, short, mousebite, spur, pin hole, spurious copper)."
        ),
        "num_classes": 6,

        # IMPORTANT:
        # YOLO labels for DeepPCB should use this class ordering (0..5).
        "class_names": [
            "open",             # 0
            "short",            # 1
            "mousebite",        # 2
            "spur",             # 3
            "pin_hole",         # 4
            "spurious_copper",  # 5
        ],

        # YOLOv8 data.yaml for DeepPCB (relative to project root)
        "yolo_data_yaml": "data/yolo/deeppcb/data.yaml",

        # CNN dataset root for DeepPCB
        #   data/cnn/deeppcb/train/<class_name>/*.png
        #   data/cnn/deeppcb/val/<class_name>/*.png
        #   data/cnn/deeppcb/test/<class_name>/*.png
        "cnn_data_root": "data/cnn/deeppcb",
    },

    "kaggle_pcb": {
        "description": (
            "Kaggle / PKU PCB Defects: ~1386 RGB images with 6 defect types "
            "(missing hole, mouse bite, open circuit, short, spur, spurious copper)."
        ),
        "num_classes": 6,

        # YOLO labels for Kaggle PCB should use this class ordering (0..5).
        "class_names": [
            "missing_hole",     # 0
            "mouse_bite",       # 1
            "open_circuit",     # 2
            "short",            # 3
            "spur",             # 4
            "spurious_copper",  # 5
        ],

        # YOLOv8 data.yaml for Kaggle PCB
        "yolo_data_yaml": "data/yolo/kaggle_pcb/data.yaml",

        # CNN dataset root for Kaggle PCB
        #   data/cnn/kaggle_pcb/train/<class_name>/*.png
        #   data/cnn/kaggle_pcb/val/<class_name>/*.png
        #   data/cnn/kaggle_pcb/test/<class_name>/*.png
        "cnn_data_root": "data/cnn/kaggle_pcb",
    },
}
