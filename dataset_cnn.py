"""
    ======================================================================
                              Dataset CNN Script
    ======================================================================
    Name:          dataset_cnn.py
    Author:        Luis Obed Vega Maisonet
    University:    University of Central Florida (UCF)
    Course:        CAP 5415 - Computer Vision
    Academic Year: 2025
    Semester:      Fall
    Professor:     Dr. Yogesh Singh Rawat

    Description:
    This script defines the custom dataset class used by the CNN side.
    The class PCBCNNDataset takes a root directory and a split name
    (train, val, or test) and expects a folder-per-class structure
    beneath that split. It walks the directory to discover class names
    from subfolder names, maps them to integer labels, and collects all
    image file paths with their corresponding labels. When the dataloader
    asks for a sample, the class loads the image with Pillow, converts it
    to RGB, applies any transforms you passed in (for resizing,
    normalization, augmentation, and so on), and returns the transformed
    image tensor together with its class index. This script is what makes
    the CNN training code independent of a specific dataset layout and
    gives a clean interface to read images and labels.
------------------------------------------------Imports and Globals-----------------------------------------------------
"""
import os
from typing import Callable, List, Tuple
from PIL import Image
from torch.utils.data import Dataset
"""-----------------------------------------------Class Definition---------------------------------------------------"""
class PCBCNNDataset(Dataset):
    """-------------------------------------------------Functions----------------------------------------------------"""
    """
        Function Name: __init__
        Description:
            Initializes the PCBCNNDataset instance for a specific split of the
            dataset. This constructor validates the requested split ('train',
            'val', or 'test'), locates the corresponding split directory under
            the given root directory, discovers all class subfolders, assigns
            each class a unique integer index, and collects every valid image
            file path together with its class label into an internal list of
            samples. If no images are found for the split, it raises a
            RuntimeError to alert the caller that the dataset is empty or the
            path is misconfigured.
        Input:
            root_dir (str):
                Path to the dataset root directory that contains the 'train',
                'val', and 'test' subdirectories.
            split (str, optional):
                Name of the split to load. Must be one of 'train', 'val', or
                'test'. The default value is 'train'.
            transform (Callable, optional):
                Optional transformation function applied to each loaded image.
                This is typically a torchvision transform pipeline that
                performs resizing, normalization, and/or data augmentation
                before the image is returned.
        Output:
            None.
    """
    def __init__(self,
                 root_dir: str,
                 split: str = "train",
                 transform: Callable = None) -> None:
        super().__init__()
        assert split in ["train", "val", "test"], "split must be 'train', 'val', or 'test'"

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Discover class subfolders and assign integer class indices
        class_names = sorted(
            [d for d in os.listdir(split_dir)
             if os.path.isdir(os.path.join(split_dir, d))]
        )
        for idx, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name

        # Collect all image paths and their labels
        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    img_path = os.path.join(class_dir, fname)
                    label = self.class_to_idx[class_name]
                    self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found for split={split} in {split_dir}")
    """
        Function Name: __len__
        Description:
            Returns the total number of samples available in the dataset for
            the selected split. This allows PyTorch's DataLoader and other
            utilities to query the size of the dataset when creating batches
            and iterating over the data.
        Input:
            None.
        Output:
            length (int):
                Integer representing the number of (image_path, label) pairs
                stored in self.samples for the current split.
    """
    def __len__(self) -> int:
        return len(self.samples)
    """
        Function Name: __getitem__
        Description:
            Retrieves a single sample from the dataset at the given index.
            This method loads the corresponding image file from disk using
            Pillow, converts it to RGB, applies the optional transform (if
            provided), and returns the processed image together with its
            integer class label. It is the core method that enables indexed
            access to samples when using a PyTorch DataLoader.
        Input:
            idx (int):
                Zero-based index of the sample to retrieve from the dataset.
                Must be in the range [0, len(self) - 1].
        Output:
            img, label:
                img (Any):
                    Transformed image object, typically a PyTorch tensor if a
                    torchvision transform pipeline is used.
                label (int):
                    Integer class index corresponding to the defect category
                    of the image located at the specified index.
    """
    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label