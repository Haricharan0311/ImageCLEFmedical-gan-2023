"""
General utilities
"""
import copy

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import Tensor, nn


def sliding_average(value_list, window):
    """
    Computes the average of the latest instances in a list
    """

    if len(value_list) == 0:
        raise ValueError("Cannot perform sliding average on an empty list.")
    return np.asarray(value_list[-window:]).mean()


def compute_backbone_output_shape(backbone):

    """
    Compute the dimension of the feature space defined by a feature extractor.
    """

    input_images = torch.ones((4, 3, 32, 32))
    output = copy.deepcopy(backbone).cpu()(input_images)
    return output.shape[1:]


def compute_prototypes(support_features, support_labels):
    """
    Compute class prototypes from support features and labels
    """

    n_way = len(torch.unique(support_labels))
    # Prototype i is the mean of all instances of features corresponding to labels == i
    return torch.cat(
        [
            support_features[torch.nonzero(support_labels == label)].mean(0)
            for label in range(n_way)
        ]
    )


def entropy(logits):
    """
    Compute entropy of prediction.
    NOTE: Takes logit as input, not probability.
    """

    probabilities = logits.softmax(dim=1)
    return (-(probabilities * (probabilities + 1e-12).log()).sum(dim=1)).mean()