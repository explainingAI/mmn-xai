# -*- coding: utf-8 -*-
""" Module for the Out of Domain inconsistent behavior detector.

Out of Domain Inconsistent Behavior Detector (OOD-IBD) is a tool for detecting bad output of a model
based out-of-domain (OOD) data. The definition of what is considered an inconsistent output is done
with the difference between the output of the model and the ideal output.

Written by Miquel Miró-Nicolau (UIB), 2022.

"""
from typing import Callable, Union

__all__ = ['detect']


def detect(model: Callable, extreme_cases: list, threshold: float = None,
           verbose: bool = False) -> Union[bool, float, int]:
    """ Automatic method to detect inconsistent results for out of domain samples.

    Args:
        model: Function returning a tensor (numpy array) of
        extreme_cases: List of extreme inputs.
        threshold: Level of difference acceptable to not be considered inconsistent. Default a 25
            per cent of the maximum possible difference.
        verbose: Flag. If true returns the number of extreme cases with inconsistent results.

    Returns:

    """
    ideal_value = None
    is_ood = 0

    for extrem_c in extreme_cases:
        res = model(extrem_c)

        if ideal_value is None:
            ideal_value = 1 / len(res)

            if threshold is None:
                threshold = (1 - ideal_value) * 0.25

        is_ood += int((max(res) - ideal_value) > threshold)

    if not verbose:
        is_ood = bool(is_ood)

    return is_ood
