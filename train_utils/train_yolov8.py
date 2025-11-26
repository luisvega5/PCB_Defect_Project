"""
    ======================================================================
                              Train Yolov8 Script
    ======================================================================
    Name:          train_yolov8.py
    Author:        Luis Obed Vega Maisonet
    University:    University of Central Florida (UCF)
    Course:        CAP 5415 - Computer Vision
    Academic Year: 2025
    Semester:      Fall
    Professor:     Dr. Yogesh Singh Rawat

    Description:
    This script acts as a wrapper around the Ultralytics YOLOv8 library.
    It is responsible for training and evaluating a YOLOv8 object
    detection model on a dataset described by a YOLO-style data.yaml file.
    Given the path to that YAML, a set of hyperparameters (such as number
    of epochs and image size), and some naming information for runs, it
    loads a YOLOv8 model (for example yolov8n.pt), calls the built-in
    model.train method to train on the specified dataset, and then calls
    model.val to evaluate on the validation or test split. The script
    extracts key detection metrics (such as mAP@0.5 and mAP@0.5:0.95) from
    the Ultralytics results object and returns them in a simple dictionary.
    All training artifacts, including the best weights, training curves,
    confusion matrices, and precision–recall plots, are written by YOLO
    itself into a run folder under runs_yolov8.
------------------------------------------------Imports and Globals-----------------------------------------------------
"""
from typing import Dict, Optional
from ultralytics import YOLO, settings
# ----------------------------------------------------------------------
# YOLO modification to run offline:
# Done by making sync=False since it disables analytics + crash reporting
# ----------------------------------------------------------------------
settings.update({"sync": False})
from utils.seed_utils import set_global_seed
"""---------------------------------------------------Functions------------------------------------------------------"""
"""
    Function Name: train_and_evaluate_yolov8
    Description:
        Trains a YOLOv8 object detection model for PCB defect detection using
        the Ultralytics library and evaluates it on the validation/test split
        defined in a YOLO-format data.yaml file. This function first sets the
        global random seed for reproducibility, then loads a YOLOv8 model
        using either a model name (e.g., "yolov8n.pt") or a local weights
        file. It calls model.train() with the specified training parameters
        (data.yaml path, number of epochs, image size, project directory, and
        run name), after which it calls model.val() to compute detection
        metrics on the evaluation set. The function extracts key metrics such
        as mAP@0.5 and mAP@0.5:0.95 from the Ultralytics metrics object and
        returns them in a summary dictionary along with run-identifying
        information. All training artifacts and plots are written by YOLO
        into the designated project/run folders.
    Input:
        data_yaml (str, optional):
            Path to the YOLO-format dataset configuration file (data.yaml)
            describing train/val/test image and label locations. Default is
            "data/yolo/data.yaml".
        model_weights (str, optional):
            Path to a YOLOv8 weights file or a model name recognized by the
            Ultralytics library (e.g., "yolov8n.pt"). Default is "yolov8n.pt".
        epochs (int, optional):
            Number of training epochs to run. Default is 50.
        imgsz (int, optional):
            Image size (input resolution) used during training and evaluation.
            YOLO will resize images so that the longer side is approximately
            this size. Default is 640.
        project (str, optional):
            Root directory where YOLO will create its run folders (e.g.,
            project/name). Default is "runs_yolov8".
        name (str, optional):
            Name of the specific training run inside the project directory.
            The evaluation run will use this name with "_eval" appended.
            Default is "yolov8_pcb".
        seed (int, optional):
            Random seed used for reproducible experiments. It is passed to
            set_global_seed() before training. Default is 42.
    Output:
        summary (Dict):
            Dictionary containing a compact summary of YOLO training and
            evaluation results, including:
                "map50":        mAP@0.5 for bounding box predictions (float or
                                None if not available).
                "map50_95":     mAP@0.5:0.95 for bounding box predictions
                                (float or None if not available).
                "project":      project directory where runs are stored (str).
                "train_run_name": name of the training run folder (str).
                "eval_run_name":  name of the evaluation run folder (str).
"""
def train_and_evaluate_yolov8(
    data_yaml: str = "data/yolo/data.yaml",
    model_weights: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    project: str = "runs_yolov8",
    name: str = "yolov8_pcb",
    seed: int = 42,
) -> Dict:

    # Set seed for reproducibility
    set_global_seed(seed)

    # Load YOLOv8 model (can be a model name or path to .pt file)
    model = YOLO(model_weights)

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=project,
        name=name,
        verbose=True,
    )

    # Evaluate the model (validation / test, depending on data.yaml)
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        project=project,
        name=f"{name}_eval",
    )

    # Extract some useful metrics (mAP, etc.)
    # NOTE: 'metrics' is an Ultralytics object; here we extract common fields.
    summary = {
        "map50": getattr(metrics, "box", None).map50 if hasattr(metrics, "box") else None,
        "map50_95": getattr(metrics, "box", None).map if hasattr(metrics, "box") else None,
        "project": project,
        "train_run_name": name,
        "eval_run_name": f"{name}_eval",
    }

    print("[YOLOv8] Training and evaluation completed.")
    print(f"[YOLOv8] mAP@0.5: {summary['map50']}")
    print(f"[YOLOv8] mAP@0.5:0.95: {summary['map50_95']}")

    return summary