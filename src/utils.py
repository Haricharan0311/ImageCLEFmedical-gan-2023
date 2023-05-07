"""
General utilities
"""
import copy

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torcheval.aucs.aggregation.auc import AUC
from tqdm import tqdm


def sliding_average(value_list, window):
    """
    Computes the average of the latest instances in a list
    """

    if len(value_list) == 0:
        raise ValueError("Cannot perform sliding average on an empty list.")
    return np.asarray(value_list[-window:]).mean()


def compute_backbone_output_shape(backbone, in_channels=1):

    """
    Compute the dimension of the feature space defined by a feature extractor.
    """

    input_images = torch.ones((4, in_channels, 32, 32))
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


def evaluate_on_one_task(
    model,
    imgs_real,
	imgs_generated,
	similarity_scores
):
    """
    Returns the number of correct predictions of query labels, and the total number of
    predictions.
    """
    predictions = model(imgs_real, imgs_generated).detach().data

    auc = AUC()
    auc.update(predictions, similarity_scores)
    auc_value = auc.compute()
    auc.reset()

    acc = BinaryAccuracy(threshold=0.5)
    acc.update(predictions, similarity_scores)
    acc_val = acc.compute()
    acc.reset()

    return auc_val, acc_val


def evaluate(
    model,
    data_loader,
    device = "cuda",
    use_tqdm = True,
    tqdm_prefix = None,
):
    """
    Evaluate the model on few-shot classification tasks
    """
    
    auc = AUC()
    auc.update(predictions, similarity_scores)
    auc_value = auc.compute()

    acc = BinaryAccuracy(threshold=0.5)
    acc.update(predictions, similarity_scores)

    model.eval()
    with torch.no_grad():
        with tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            disable=not use_tqdm,
            desc=tqdm_prefix,
        ) as tqdm_eval:
            for _, (
                imgs_real,
				imgs_generated,
				similarity_scores 
            ) in tqdm_eval:
                
                predictions = model(imgs_real.to(device), imgs_generated.to(device)).detach().data
                auc.update(predictions, similarity_scores.to(device))
                acc.update(predictions, similarity_scores.to(device))

                # Log accuracy in real time
                tqdm_eval.set_postfix(accuracy=acc.compute(), auc=auc.compute())

    return auc.compute(), acc.compute()