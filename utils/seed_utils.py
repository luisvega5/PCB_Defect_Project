"""
    ======================================================================
                            Seed Utils Script
    ======================================================================
    Name:          seed_utils.py
    Author:        Luis Obed Vega Maisonet
    University:    University of Central Florida (UCF)
    Course:        CAP 5415 - Computer Vision
    Academic Year: 2025
    Semester:      Fall
    Professor:     Dr. Yogesh Singh Rawat

    Description:
    This script contains a small utility function to make your experiments
    reproducible. The function set_global_seed takes an integer seed and
    sets that seed for the Python random module, NumPy, and PyTorch
    (both CPU and GPU). It also configures the CuDNN backend to run in
    deterministic mode and disables certain optimizations that can
    introduce nondeterminism. Both train_cnn.py and train_yolov8.py call
    this function at the beginning of their training routines so that,
    as much as possible, running the same experiment twice with the same
    code and data will give you consistent results.
------------------------------------------------Imports and Globals-----------------------------------------------------
"""
import random
import numpy as np
import torch
"""---------------------------------------------------Functions------------------------------------------------------"""
"""
    Function Name: set_global_seed
    Description:
        Sets a global random seed for all major randomness sources used in the
        project to help make experiments reproducible. This function seeds the
        Python built-in random module, NumPy, and PyTorch (for both CPU and GPU
        computations). It also configures the PyTorch CuDNN backend to run in
        deterministic mode and disables certain benchmarking optimizations that
        can introduce non-deterministic behavior between runs.
    Input:
        seed (int, optional):
            Seed value used to initialize all random number generators. The
            default is 42, but any integer can be provided to reproduce a
            specific experiment.
    Output:
        None.
        This function does not return a value. Its effect is to configure the
        global random state for Python, NumPy, and PyTorch so that subsequent
        training and evaluation code behaves as deterministically as possible.
"""
def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full reproducibility (can slow things down slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False