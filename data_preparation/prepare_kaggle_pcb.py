"""
    ======================================================================
                            Prepare Kaggle PCB Script
    ======================================================================
    Name:          prepare_kaggle_pcb.py
    Author:        Luis Obed Vega Maisonet
    University:    University of Central Florida (UCF)
    Course:        CAP 5415 - Computer Vision
    Academic Year: 2025
    Semester:      Fall
    Professor:     Dr. Yogesh Singh Rawat

    Description:
    This script plays the same role as prepare_deeppcb.py but for the
    Kaggle PCB Defects dataset. It starts from the raw Kaggle folder
    structure, typically with images organized in class-specific subfolders
    and annotations stored as VOC-style XML files. The script normalizes
    class names to a consistent set of six defect classes, collects all
    image–annotation pairs, and again assigns each example to a train,
    validation, or test split according to specified ratios. For the
    YOLO dataset, it copies full images into
    data/yolo/kaggle_pcb/images/<split>/ and converts annotation
    bounding boxes into YOLO label files under
    data/yolo/kaggle_pcb/labels/<split>/. For the CNN dataset,
    it performs full-image classification by copying each image
    into data/cnn/kaggle_pcb/<split>/<class_name>/ based on its defect
    type. The end result is that the Kaggle dataset is available in
    both detection and classification formats, aligned with the rest
    of the pipeline.
------------------------------------------------Imports and Globals-----------------------------------------------------
"""
import os
import random
import shutil
import xml.etree.ElementTree as ET
# Raw Kaggle PCB Defects dataset
RAW_ROOT = "data/raw/kaggle_pcb_defects"
IMG_ROOT = os.path.join(RAW_ROOT, "images")
ANN_ROOT = os.path.join(RAW_ROOT, "Annotations")

# Output roots
YOLO_OUT_ROOT = "data/yolo/kaggle_pcb"
CNN_OUT_ROOT = "data/cnn/kaggle_pcb"

# Canonical class order for both YOLO and CNN (match data.yaml)
CLASS_NAME_ORDER = [
    "missing_hole",     # 0
    "mouse_bite",       # 1
    "open_circuit",     # 2
    "short",            # 3
    "spur",             # 4
    "spurious_copper",  # 5
]

CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAME_ORDER)}

SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}
"""---------------------------------------------------Functions------------------------------------------------------"""
"""
    Function Name: norm_name
    Description:
        Normalizes a raw class or object name string into a canonical form
        used throughout the script. The function converts the string to
        lowercase, strips leading and trailing whitespace, and replaces any
        internal spaces with underscores so that different naming styles
        map to a consistent identifier.
    Input:
        name (str):
            Raw name string to normalize, typically read from folder names
            or XML annotation tags.
    Output:
        normalized (str):
            Normalized version of the input string in lowercase with spaces
            replaced by underscores.
"""
def norm_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")
"""
    Function Name: collect_records
    Description:
        Walks the Kaggle PCB Defects images directory tree and collects all
        valid image–annotation pairs along with their canonical class names.
        For each image file under IMG_ROOT whose class folder corresponds to
        a known defect category and that has a matching VOC-style XML file
        in the Annotations directory, this function records the full image
        path, XML path, and normalized class label in a dictionary.
    Input:
        None.
        The function relies on the global IMG_ROOT and ANN_ROOT paths to
        locate images and annotations on disk.
    Output:
        records (List[Dict[str, str]]):
            A list of dictionaries, each containing:
                "img_path": absolute path to the image file,
                "xml_path": absolute path to the corresponding XML file,
                "class_name": canonical defect class name derived from the
                              image folder.
"""
def collect_records():
    records = []

    for root, dirs, files in os.walk(IMG_ROOT):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(root, fname)
            rel = os.path.relpath(img_path, IMG_ROOT)  # e.g. Missing_hole/01_...jpg
            parts = rel.split(os.sep)
            if len(parts) < 2:
                continue

            folder_raw = parts[0]
            cls = norm_name(folder_raw)

            # Some variants use "open" instead of "open_circuit"
            if cls == "open":
                cls = "open_circuit"

            if cls not in CLASS_NAME_TO_ID:
                print("Skipping unknown class folder", folder_raw, "for", img_path)
                continue

            xml_rel = os.path.splitext(rel)[0] + ".xml"
            xml_path = os.path.join(ANN_ROOT, xml_rel)
            if not os.path.exists(xml_path):
                print("Warning: missing annotation for", img_path)
                continue

            records.append(
                {
                    "img_path": img_path,
                    "xml_path": xml_path,
                    "class_name": cls,
                }
            )

    return records
"""
    Function Name: split_indices
    Description:
        Randomly partitions a sequence of sample indices into train,
        validation, and test subsets according to the SPLIT_RATIOS
        configuration. The resulting index sets are later used to ensure
        that each image–annotation pair is assigned consistently to exactly
        one split across both the YOLO and CNN datasets.
    Input:
        n (int):
            Total number of records, typically len(records) where records
            is the list returned by collect_records().
    Output:
        train_idx, val_idx, test_idx (Tuple[Set[int], Set[int], Set[int]]):
            Three disjoint sets of integer indices representing the train,
            validation, and test splits. Their union covers the full range
            [0, n - 1].
"""
def split_indices(n):
    idxs = list(range(n))
    random.shuffle(idxs)

    n_train = int(SPLIT_RATIOS["train"] * n)
    n_val = int(SPLIT_RATIOS["val"] * n)

    train_idx = set(idxs[:n_train])
    val_idx = set(idxs[n_train:n_train + n_val])
    test_idx = set(idxs[n_train + n_val:])

    return train_idx, val_idx, test_idx
"""
    Function Name: ensure_dirs
    Description:
        Creates the required directory structure for both the YOLOv8 and
        CNN versions of the Kaggle PCB dataset. For each split ('train',
        'val', 'test'), this function ensures that the corresponding YOLO
        images and labels folders exist under YOLO_OUT_ROOT and that, for
        the CNN dataset, a subfolder is created for every defect class
        listed in CLASS_NAME_ORDER under CNN_OUT_ROOT.
    Input:
        None.
    Output:
        None.
        The function does not return a value. Its effect is to create all
        necessary output directories if they do not already exist.
"""
def ensure_dirs():
    for split in ["train", "val", "test"]:
        # YOLO dirs
        os.makedirs(os.path.join(YOLO_OUT_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_OUT_ROOT, "labels", split), exist_ok=True)

        # CNN dirs per class
        for cls in CLASS_NAME_ORDER:
            os.makedirs(os.path.join(CNN_OUT_ROOT, split, cls), exist_ok=True)
"""
    Function Name: parse_boxes
    Description:
        Parses a VOC-style XML annotation file and extracts image size
        information along with all annotated bounding boxes. For each
        <object> tag in the XML, the function reads the defect name and
        bounding box coordinates, normalizes the name using norm_name, and
        returns the list of boxes together with the image width and height.
    Input:
        xml_path (str):
            Path to the XML annotation file associated with a particular
            Kaggle PCB image.
    Output:
        W, H, boxes:
            W (int):
                Image width in pixels as specified in the XML.
            H (int):
                Image height in pixels as specified in the XML.
            boxes (List[Tuple[str, int, int, int, int]]):
                List of tuples (name, xmin, ymin, xmax, ymax) for each
                annotated defect region in the image, where name is the
                raw (still normalized) class name and the coordinates are
                integer pixel positions.
"""
def parse_boxes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    W = int(size.find("width").text)
    H = int(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        name = norm_name(obj.find("name").text)
        bnd = obj.find("bndbox")
        xmin = int(float(bnd.find("xmin").text))
        ymin = int(float(bnd.find("ymin").text))
        xmax = int(float(bnd.find("xmax").text))
        ymax = int(float(bnd.find("ymax").text))
        boxes.append((name, xmin, ymin, xmax, ymax))

    return W, H, boxes
"""
    Function Name: process_record
    Description:
        Processes a single image–annotation record and writes it into both
        the YOLOv8 and CNN dataset structures for a given split. For the
        YOLO branch, it parses the XML file to obtain image size and
        bounding boxes, copies the original image into the split-specific
        images folder, and writes a YOLO-format label file with normalized
        coordinates and class IDs. For the CNN branch, it performs
        full-image classification by copying the entire image into the
        corresponding class subfolder under the CNN output root.
    Input:
        rec (Dict[str, str]):
            A record dictionary produced by collect_records() containing:
                "img_path": path to the image file,
                "xml_path": path to the XML annotation file,
                "class_name": canonical defect class name for the image.
        split (str):
            Name of the data split ('train', 'val', or 'test') to which this
            record has been assigned.
    Output:
        None.
        The function creates the appropriate image and label files on disk
        for both YOLO and CNN outputs but does not return a value.
"""
def process_record(rec, split):
    img_path = rec["img_path"]
    xml_path = rec["xml_path"]
    class_name = rec["class_name"]  # canonical image-level class

    stem = os.path.splitext(os.path.basename(img_path))[0]

    # ---- YOLO branch ----
    W, H, boxes = parse_boxes(xml_path)

    out_img = os.path.join(YOLO_OUT_ROOT, "images", split, stem + ".jpg")
    shutil.copy2(img_path, out_img)

    out_lab = os.path.join(YOLO_OUT_ROOT, "labels", split, stem + ".txt")
    yolo_lines = []

    for (name, xmin, ymin, xmax, ymax) in boxes:
        cname = norm_name(name)

        # Normalize "open" naming if necessary
        if cname == "open":
            cname = "open_circuit"

        if cname not in CLASS_NAME_TO_ID:
            # Fall back to the image-level class
            cname = class_name

        cid = CLASS_NAME_TO_ID[cname]

        xmin = max(0, min(W - 1, xmin))
        xmax = max(0, min(W, xmax))
        ymin = max(0, min(H - 1, ymin))
        ymax = max(0, min(H, ymax))

        if xmax <= xmin or ymax <= ymin:
            continue

        xc = (xmin + xmax) / 2.0 / W
        yc = (ymin + ymax) / 2.0 / H
        bw = (xmax - xmin) / W
        bh = (ymax - ymin) / H

        yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    with open(out_lab, "w") as f:
        f.write("\n".join(yolo_lines))

    # ---- CNN branch (full-image classification) ----
    cnn_dest = os.path.join(CNN_OUT_ROOT, split, class_name, stem + ".jpg")
    shutil.copy2(img_path, cnn_dest)
"""
    Function Name: main
    Description:
        Main entry point for the Kaggle PCB dataset preparation script. It
        seeds the Python random module for reproducibility, ensures that all
        output directories exist, then collects all valid image–annotation
        records from the raw Kaggle dataset. It computes a train/validation/
        test split over these records, and for each record calls
        process_record() to generate both YOLOv8 detection data and CNN
        classification data in the appropriate split. At the end, it prints
        a short summary indicating how many images were found and the final
        locations of the generated YOLO and CNN datasets.
    Input:
        None.
        All configuration information is taken from global constants such as
        RAW_ROOT, YOLO_OUT_ROOT, CNN_OUT_ROOT, CLASS_NAME_ORDER, and
        SPLIT_RATIOS.
    Output:
        None.
        The function does not return a value. Its side effects are to build
        the prepared Kaggle PCB datasets under data/yolo/kaggle_pcb and
        data/cnn/kaggle_pcb and to print status messages to the console.
"""
def main():
    random.seed(42)
    ensure_dirs()

    records = collect_records()
    print(f"Found {len(records)} Kaggle PCB images with annotations")

    train_idx, val_idx, test_idx = split_indices(len(records))

    for idx, rec in enumerate(records):
        if idx in train_idx:
            split = "train"
        elif idx in val_idx:
            split = "val"
        else:
            split = "test"

        process_record(rec, split)

    print("Done.")
    print("YOLOv8 dataset at:", YOLO_OUT_ROOT)
    print("CNN dataset at   :", CNN_OUT_ROOT)

# Runs the main function at the start of execution
if __name__ == "__main__":
    main()
