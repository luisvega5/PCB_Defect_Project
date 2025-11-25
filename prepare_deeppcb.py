"""
    ======================================================================
                            Prepare DeepPCB Script
    ======================================================================
    Name:          prepare_deeppcb.py
    Author:        Luis Obed Vega Maisonet
    University:    University of Central Florida (UCF)
    Course:        CAP 5415 - Computer Vision
    Academic Year: 2025
    Semester:      Fall
    Professor:     Dr. Yogesh Singh Rawat

    Description:
    This script is a preprocessing tool that converts the original DeepPCB
    dataset into the two formats: a YOLOv8-style detection dataset and
    a CNN-style classification dataset. It reads the raw DeepPCB directory
    under data/raw/DeepPCB-master/PCBData, finds each tested image
    (*_test.jpg) that has an associated annotation text file, and parses
    the bounding box coordinates and defect type IDs from those labels.
    It then randomly assigns each image to the train, validation, or
    test split using fixed ratios and uses that split consistently for
    both output formats. For the YOLO part, it saves the full defective
    boards into data/yolo/deeppcb/images/<split>/ and writes corresponding
    YOLO label files with normalized coordinates into
    data/yolo/deeppcb/labels/<split>/. For the CNN part, it crops each
    defect bounding box from the tested image, and writes these patches
    into data/cnn/deeppcb/<split>/<class_name>/ according to defect type.
    This script transforms the original dataset into ready-to-use training
    data for both detection and classification models.
------------------------------------------------Imports and Globals-----------------------------------------------------
"""
import os
import random
from typing import List, Tuple
from PIL import Image
# Path to original DeepPCB data (as in DeepPCB-master/PCBData)
RAW_ROOT = "data/raw/DeepPCB-master/PCBData"

# Output roots
CNN_OUT_ROOT = "data/cnn/deeppcb"
YOLO_OUT_ROOT = "data/yolo/deeppcb"

# Mapping from DeepPCB label IDs (1..6) to class names
CLASS_ID_TO_NAME = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "spurious_copper",  # DeepPCB calls this 'copper'
    6: "pin_hole",
}

# Train/val/test split ratios (over the 1500 images)
SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}
"""---------------------------------------------------Functions------------------------------------------------------"""
"""
    Function Name: collect_samples
    Description:
        Walks the original DeepPCB PCBData directory tree and collects all
        valid image–annotation pairs. For every tested image file whose name
        ends with '_test.jpg' and that has a corresponding annotation text
        file in the matching '*_not' folder, this function records the full
        paths to the image and label files together with the common stem
        (ID). Only numeric subfolders like '00041' are considered as valid
        sample groups.
    Input:
        None.
        The function uses the global RAW_ROOT path to locate the original
        DeepPCB dataset on disk.
    Output:
        samples (List[Tuple[str, str, str]]):
            A list of tuples of the form (img_path, label_path, stem), where
            img_path is the absolute path to a '*_test.jpg' image, label_path
            is the absolute path to its corresponding '.txt' annotation file,
            and stem is the shared base name used to link the two.
"""
def collect_samples() -> List[Tuple[str, str, str]]:
    samples = []
    for group in sorted(os.listdir(RAW_ROOT)):
        gpath = os.path.join(RAW_ROOT, group)
        if not os.path.isdir(gpath):
            continue

        for sub in sorted(os.listdir(gpath)):
            spath = os.path.join(gpath, sub)
            # We only care about numeric subfolders like "00041"
            if not os.path.isdir(spath) or not sub.isdigit():
                continue

            img_dir = spath
            lab_dir = spath + "_not"
            if not os.path.isdir(lab_dir):
                continue

            for fname in os.listdir(img_dir):
                if not fname.endswith("_test.jpg"):
                    continue

                stem = fname.replace("_test.jpg", "")
                img_path = os.path.join(img_dir, fname)
                lab_path = os.path.join(lab_dir, stem + ".txt")
                if not os.path.exists(lab_path):
                    continue

                samples.append((img_path, lab_path, stem))
    return samples
"""
    Function Name: split_indices
    Description:
        Randomly partitions a range of sample indices into train, validation,
        and test subsets according to the global SPLIT_RATIOS. The same
        partitioning is later used for both the YOLO and CNN datasets so that
        each original image belongs to exactly one split consistently across
        both formats.
    Input:
        n (int):
            Total number of available samples, typically len(samples) where
            samples is the list returned by collect_samples().
    Output:
        train_idx, val_idx, test_idx (Tuple[Set[int], Set[int], Set[int]]):
            Three sets containing the integer indices assigned to the train,
            validation, and test splits respectively. The sets are disjoint
            and their union covers the range [0, n - 1].
"""
def split_indices(n: int):
    idxs = list(range(n))
    random.shuffle(idxs)

    n_train = int(SPLIT_RATIOS["train"] * n)
    n_val = int(SPLIT_RATIOS["val"] * n)

    train_idx = set(idxs[:n_train])
    val_idx = set(idxs[n_train:n_train + n_val])
    test_idx = set(idxs[n_train + n_val:])

    return train_idx, val_idx, test_idx
"""
    Function Name: parse_label_file
    Description:
        Parses a DeepPCB annotation text file and converts it into a list of
        cleaned bounding boxes. Each line in the label file is expected to
        contain 'x1,y1,x2,y2,type' (or the same values separated by
        whitespace). The function clamps box coordinates to the image bounds,
        filters out invalid or degenerate boxes, and ignores any label whose
        class ID is outside the valid range of 1–6.
    Input:
        label_path (str):
            Path to the DeepPCB annotation file corresponding to a tested
            image.
        W (int):
            Width of the associated image in pixels.
        H (int):
            Height of the associated image in pixels.
    Output:
        boxes (List[Tuple[int, int, int, int, int]]):
            A list of bounding boxes, where each element is a tuple
            (x1, y1, x2, y2, cid). The coordinates are clamped to [0, W] and
            [0, H] respectively, and cid is an integer class ID in the range
            1–6 representing one of the DeepPCB defect categories.
"""
def parse_label_file(label_path: str, W: int, H: int):
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Original DeepPCB uses "x1,y1,x2,y2,type" (comma-separated)
            parts = line.split(",")
            if len(parts) == 1:
                # Fallback in case of whitespace-separated
                parts = line.split()

            if len(parts) != 5:
                continue

            x1, y1, x2, y2, cid = map(int, parts)

            if cid < 1 or cid > 6:
                # 0 is background, only keep 1..6
                continue

            # Clamp coordinates to image bounds
            x1 = max(0, min(W - 1, x1))
            x2 = max(0, min(W, x2))
            y1 = max(0, min(H - 1, y1))
            y2 = max(0, min(H, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append((x1, y1, x2, y2, cid))

    return boxes
"""
    Function Name: ensure_yolo_dirs
    Description:
        Ensures that the directory structure required for the YOLOv8 dataset
        exists. For each split ('train', 'val', 'test'), this function creates
        the corresponding 'images' and 'labels' subdirectories under the
        YOLO_OUT_ROOT path if they do not already exist.
    Input:
        None.
    Output:
        None.
        The function does not return a value. Its effect is to create the
        required folder hierarchy for saving YOLO images and label files.
"""
def ensure_yolo_dirs():
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(YOLO_OUT_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_OUT_ROOT, "labels", split), exist_ok=True)
"""
    Function Name: ensure_cnn_dirs
    Description:
        Ensures that the base directory structure required for the CNN
        dataset exists. For each split ('train', 'val', 'test'), this function
        creates the corresponding subdirectory under the CNN_OUT_ROOT path if
        it does not already exist. Class-specific folders are created later
        when individual patches are saved.
    Input:
        None.
    Output:
        None.
        The function does not return a value. Its effect is to create the base
        split directories for the CNN patch dataset.
"""
def ensure_cnn_dirs():
    # We can create per-class dirs lazily, but this ensures base splits exist.
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(CNN_OUT_ROOT, split), exist_ok=True)
"""
    Function Name: save_yolo_example
    Description:
        Saves a single DeepPCB sample in YOLOv8 format. The function writes
        the full defective board image to the appropriate split-specific
        'images' folder and creates a corresponding YOLO label file in the
        'labels' folder. Each DeepPCB bounding box is converted into YOLO
        format (class_id, x_center_norm, y_center_norm, width_norm,
        height_norm), where class IDs are mapped from the original 1–6 range
        to 0–5.
    Input:
        img (PIL.Image.Image):
            Loaded RGB image corresponding to a tested PCB (the '*_test.jpg'
            file).
        W (int):
            Width of the image in pixels.
        H (int):
            Height of the image in pixels.
        boxes (Iterable[Tuple[int, int, int, int, int]]):
            Iterable of bounding boxes as (x1, y1, x2, y2, cid), typically
            returned by parse_label_file().
        stem (str):
            Base name used to generate the output file names (without
            extension).
        split (str):
            Name of the data split ('train', 'val', or 'test') where this
            sample should be saved.
    Output:
        None.
        The function writes an image file and a label text file to disk but
        does not return a value.
"""
def save_yolo_example(img: Image.Image,
                      W: int,
                      H: int,
                      boxes,
                      stem: str,
                      split: str):
    # Save one DeepPCB sample as YOLO-style image + label file.
    out_img = os.path.join(YOLO_OUT_ROOT, "images", split, stem + ".jpg")
    img.convert("RGB").save(out_img)

    out_lab = os.path.join(YOLO_OUT_ROOT, "labels", split, stem + ".txt")
    yolo_lines = []

    for (x1, y1, x2, y2, cid) in boxes:
        # Map 1..6 -> 0..5
        class_id = cid - 1

        xc = (x1 + x2) / 2.0 / W
        yc = (y1 + y2) / 2.0 / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H

        yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    with open(out_lab, "w") as f:
        f.write("\n".join(yolo_lines))
"""
    Function Name: save_cnn_patches
    Description:
        Saves cropped defect patches from a single DeepPCB sample for use in
        the CNN classification dataset. For each bounding box, the function
        crops the corresponding region from the tested image and writes it to
        a class-specific folder under data/cnn/deeppcb/<split>/<class_name>/.
        Class names are determined from the CLASS_ID_TO_NAME mapping based on
        the DeepPCB class ID for each box.
    Input:
        img (PIL.Image.Image):
            Loaded RGB image corresponding to a tested PCB (the '*_test.jpg'
            file).
        boxes (Iterable[Tuple[int, int, int, int, int]]):
            Iterable of bounding boxes as (x1, y1, x2, y2, cid), typically
            returned by parse_label_file().
        stem (str):
            Base name used to generate the output patch file names.
        split (str):
            Name of the data split ('train', 'val', or 'test') where these
            patches should be saved.
    Output:
        None.
        The function writes cropped patch images to disk in their respective
        class folders but does not return a value.
"""
def save_cnn_patches(img: Image.Image,
                     boxes,
                     stem: str,
                     split: str):
    # Save cropped defect patches from the same DeepPCB sample for CNN dataset.
    # Each patch goes into data/cnn/deeppcb/<split>/<class_name>/.
    for i, (x1, y1, x2, y2, cid) in enumerate(boxes):
        class_name = CLASS_ID_TO_NAME.get(cid)
        if class_name is None:
            continue

        patch = img.crop((x1, y1, x2, y2))
        out_dir = os.path.join(CNN_OUT_ROOT, split, class_name)
        os.makedirs(out_dir, exist_ok=True)

        out_name = f"{stem}_{i:02d}.png"
        out_path = os.path.join(out_dir, out_name)
        patch.save(out_path)
"""
    Function Name: main
    Description:
        Main entry point for the DeepPCB preparation script. This function
        seeds the random number generator for reproducibility, collects all
        valid DeepPCB image–label pairs, and computes a train/validation/test
        split over the samples according to the global SPLIT_RATIOS. It then
        ensures that the YOLO and CNN output directory structures exist and
        iterates over all samples. For each sample, it loads the tested image,
        parses its label file into bounding boxes, and, if any boxes are
        present, saves the image and labels in YOLO format and the cropped
        patches in CNN format under the appropriate split. At the end, it
        reports how many images and bounding boxes were processed and prints
        the locations of the generated datasets.
    Input:
        None.
        All configuration is taken from global constants such as RAW_ROOT,
        CNN_OUT_ROOT, YOLO_OUT_ROOT, CLASS_ID_TO_NAME, and SPLIT_RATIOS.
    Output:
        None.
        The function does not return a value. Its effect is to create fully
        prepared YOLOv8 and CNN datasets for the DeepPCB benchmark under the
        data/yolo/deeppcb and data/cnn/deeppcb directories.
"""
def main():
    random.seed(42)

    samples = collect_samples()
    print(f"Found {len(samples)} DeepPCB image+label pairs")

    train_idx, val_idx, test_idx = split_indices(len(samples))
    ensure_yolo_dirs()
    ensure_cnn_dirs()

    total_boxes = 0

    for idx, (img_path, lab_path, stem) in enumerate(samples):
        if idx in train_idx:
            split = "train"
        elif idx in val_idx:
            split = "val"
        else:
            split = "test"

        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        boxes = parse_label_file(lab_path, W, H)
        if not boxes:
            continue

        total_boxes += len(boxes)

        save_yolo_example(img, W, H, boxes, stem, split)
        save_cnn_patches(img, boxes, stem, split)

    print(f"Done. Processed {len(samples)} images and {total_boxes} boxes.")
    print("  YOLOv8 dataset under:", YOLO_OUT_ROOT)
    print("  CNN patches under  :", CNN_OUT_ROOT)

# Runs the main function at the start of execution
if __name__ == "__main__":
    main()
