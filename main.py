"""
    ======================================================================
                              Main Script
    ======================================================================
    Name:          main.py
    Author:        Luis Obed Vega Maisonet
    University:    University of Central Florida (UCF)
    Course:        CAP 5415 - Computer Vision
    Academic Year: 2025
    Semester:      Fall
    Professor:     Dr. Yogesh Singh Rawat

    Description:
    This is the main entry point for the whole project. When you run it
    from the command line, it reads your arguments (for example which
    dataset to use and whether you want to run YOLO, CNN, or both).
    It then looks up the configuration for that dataset in
    dataset_configs.py and decides which pipelines to execute.
    If you ask for YOLO, it calls the YOLO training/evaluation function in
    train_yolov8.py. If you ask for the CNN, it calls the training/evaluation
    function in train_cnn.py. When those functions finish, main.py collects
    their metrics (such as accuracy for CNN and mAP for YOLO) and writes a
    summary JSON file containing the results.
------------------------------------------------Imports and Globals-----------------------------------------------------
"""
import argparse
import json
from train_cnn import train_and_evaluate_cnn
from train_yolov8 import train_and_evaluate_yolov8
from dataset_configs import DATASET_CONFIGS
"""---------------------------------------------------Functions------------------------------------------------------"""
""" 
    Function Name: parse_args
    Description:
        Defines and parses all command-line arguments for running PCB defect
        detection experiments. It configures options to select the dataset,
        choose which model pipeline(s) to run (YOLOv8, CNN, or both), and set
        training hyperparameters for each model (e.g., epochs, image size, batch
        size, learning rate, and paths for data and checkpoints).
    Input: 
        None.
    
    Output:
        Returns an argparse.Namespace object containing all parsed command-line
        options as attributes (e.g., args.dataset, args.model, args.yolo_epochs,
        args.cnn_epochs, args.results_json, etc.).
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description="PCB Defect Detection Project - Dataset & Model Selector"
    )
    # Which dataset to use
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        default="deeppcb",
        help=(
            "Which dataset configuration to use. "
            "Options: " + ", ".join(DATASET_CONFIGS.keys())
        ),
    )
    # Which model pipeline(s) to run
    parser.add_argument(
        "--model",
        type=str,
        choices=["yolov8", "cnn", "both"],
        required=True,
        help="Which model pipeline to run: 'yolov8', 'cnn', or 'both'.",
    )
    # YOLOv8-specific arguments (can override config defaults)
    parser.add_argument(
        "--yolo-data",
        type=str,
        default="AUTO",
        help="Path to YOLOv8 data.yaml file. Use 'AUTO' to use dataset config.",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="yolov8n.pt",
        help="YOLOv8 weights or model name (e.g., 'yolov8n.pt').",
    )
    parser.add_argument(
        "--yolo-epochs",
        type=int,
        default=50,
        help="Number of epochs for YOLOv8 training.",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=640,
        help="Image size for YOLOv8 training and evaluation.",
    )
    parser.add_argument(
        "--yolo-project",
        type=str,
        default="runs_yolov8",
        help="Root directory where YOLOv8 runs will be stored.",
    )
    parser.add_argument(
        "--yolo-name",
        type=str,
        default="yolov8_pcb",
        help="Base name of the YOLOv8 run (dataset name will be appended).",
    )
    # CNN-specific arguments (can override config defaults)
    parser.add_argument(
        "--cnn-data-root",
        type=str,
        default="AUTO",
        help=(
            "Root directory for CNN dataset (folder-per-class structure). "
            "Use 'AUTO' to use dataset config."
        ),
    )
    parser.add_argument(
        "--cnn-num-classes",
        type=str,
        default="AUTO",
        help=(
            "Number of classes for CNN classification. "
            "Use 'AUTO' to use dataset config."
        ),
    )
    parser.add_argument(
        "--cnn-batch-size",
        type=int,
        default=32,
        help="Batch size for CNN training.",
    )
    parser.add_argument(
        "--cnn-epochs",
        type=int,
        default=20,
        help="Number of epochs for CNN training.",
    )
    parser.add_argument(
        "--cnn-lr",
        type=float,
        default=1e-3,
        help="Learning rate for CNN training.",
    )
    parser.add_argument(
        "--cnn-device",
        type=str,
        default=None,
        help="Device for CNN training: 'cuda', 'cpu', or None to auto-detect.",
    )
    parser.add_argument(
        "--cnn-checkpoint-dir",
        type=str,
        default="checkpoints_cnn",
        help="Directory to save CNN checkpoints.",
    )

    # Where to store a summary JSON with all metrics
    parser.add_argument(
        "--results-json",
        type=str,
        default="results_summary.json",
        help="Path to save a JSON file with collected metrics.",
    )
    args = parser.parse_args()
    return args
"""
    Function Name: main
    Description:
        It parses command-line arguments, loads the selected dataset configuration
        from DATASET_CONFIGS, resolves any parameters set to "AUTO"
        (such as YOLO data.yaml path or CNN data root), and then runs
        the requested model pipeline(s). If YOLOv8 is selected, it trains and
        evaluates the detection model and collects its metrics. If the CNN is
        selected, it trains and evaluates the classification model and collects
        its metrics. Finally, it aggregates all metrics into a dictionary and
        saves them to a JSON file for later analysis and reporting.
    Input:
        None.
    Output:
        None.
"""
def main():
    args = parse_args()
    # 1. Load dataset configuration
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{args.dataset}'. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )
    ds_cfg = DATASET_CONFIGS[args.dataset]
    print(f"\n=== Using dataset: {args.dataset} ===")
    print(ds_cfg["description"])
    print(f"Classes ({ds_cfg['num_classes']}): {ds_cfg['class_names']}")

    all_metrics = {}

    # 2. Resolve effective paths and hyperparameters from config + CLI

    # YOLOv8 data.yaml path: config vs override
    if args.yolo_data == "AUTO":
        yolo_data_yaml = ds_cfg["yolo_data_yaml"]
    else:
        yolo_data_yaml = args.yolo_data

    # CNN data root: config vs override
    if args.cnn_data_root == "AUTO":
        cnn_data_root = ds_cfg["cnn_data_root"]
    else:
        cnn_data_root = args.cnn_data_root

    # CNN number of classes: config vs override
    if args.cnn_num_classes == "AUTO":
        cnn_num_classes = ds_cfg["num_classes"]
    else:
        cnn_num_classes = int(args.cnn_num_classes)

    # 3. Run YOLOv8 pipeline, if requested
    if args.model in ["yolov8", "both"]:
        print("\n=== Running YOLOv8 pipeline ===")
        # Append dataset name to run name so runs don't clash
        yolo_run_name = f"{args.yolo_name}_{args.dataset}"

        yolo_metrics = train_and_evaluate_yolov8(
            data_yaml=yolo_data_yaml,
            model_weights=args.yolo_weights,
            epochs=args.yolo_epochs,
            imgsz=args.yolo_imgsz,
            project=args.yolo_project,
            name=yolo_run_name,
        )
        all_metrics["yolov8"] = yolo_metrics

    # 4. Run CNN pipeline, if requested
    if args.model in ["cnn", "both"]:
        print("\n=== Running CNN pipeline ===")
        cnn_metrics = train_and_evaluate_cnn(
            data_root=cnn_data_root,
            num_classes=cnn_num_classes,
            batch_size=args.cnn_batch_size,
            num_epochs=args.cnn_epochs,
            learning_rate=args.cnn_lr,
            device=args.cnn_device,
            checkpoint_dir=args.cnn_checkpoint_dir,
        )
        all_metrics["cnn"] = cnn_metrics

    # 5. Save metrics summary as JSON
    if len(all_metrics) > 0:
        with open(args.results_json, "w") as f:
            json.dump(all_metrics, f, indent=4)
        print(f"\nSaved metrics summary to {args.results_json}")

# Runs the main function at the start of execution
if __name__ == "__main__":
    main()
