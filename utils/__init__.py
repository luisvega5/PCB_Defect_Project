"""
Module Name: utils
Description:
    Utility package for shared helper functions used throughout the PCB
    Defect Detection project. This module currently re-exports the
    set_global_seed helper, which configures all major random number
    generators (Python, NumPy, and PyTorch) to improve experiment
    reproducibility. Additional utility functions can be added to this
    package over time and made available through this module.
Contents:
    set_global_seed:
        Function defined in seed_utils.py. It sets a global random seed and
        configures PyTorch backends so that training and evaluation behave
        as deterministically as possible for a given seed value.
Usage:
    from utils import set_global_seed
"""