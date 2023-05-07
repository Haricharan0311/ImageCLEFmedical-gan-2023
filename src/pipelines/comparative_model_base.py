from abc import abstractmethod

import torch
from torch import Tensor, nn

from utils import compute_backbone_output_shape, compute_prototypes


class ComparativeModelBase(nn.Module):
    """
    Abstract class providing methods usable in downstream.
    """

    def __init__(self, backbone, use_softmax):

        super().__init__()

        self.backbone = backbone
        self.backbone_output_shape = compute_backbone_output_shape(backbone)
        self.feature_dimension = self.backbone_output_shape[0]

        self.use_softmax = use_softmax

        self.prototypes = torch.tensor(())
        self.support_features = torch.tensor(())
        self.support_labels = torch.tensor(())

    
	@abstractmethod
    def forward(
        self,
        query_images
    ):
        """
        Predict classification labels.
        """
        raise NotImplementedError(
            "All few-shot algorithms must implement a forward method."
        )

    @abstractmethod
    def process_support_set(
        self,
        support_images
        support_labels
    ):
        """
        Harness information from the support set, so that query labels can later be predicted.
        """
        raise NotImplementedError(
            "All few-shot algorithms must implement a process_support_set method."
        )

    @staticmethod
    def is_transductive():
        raise NotImplementedError(
            "All few-shot algorithms must implement a is_transductive method."
        )

    def softmax_if_specified(self, output):
        """
        If the option is chosen when the classifier is initialized, we perform a softmax.
        """
        return output.softmax(-1) if self.use_softmax else output

    def l2_distance_to_prototypes(self, samples):
        """
        Compute prediction logits from their euclidean distance to support set prototypes.
        """
        return -torch.cdist(samples, self.prototypes)

    def cosine_distance_to_prototypes(self, samples):
        """
        Compute prediction logits from their cosine distance to support set prototypes.
        """
        return (
            nn.functional.normalize(samples, dim=1)
            @ nn.functional.normalize(self.prototypes, dim=1).T
        )

    def store_support_set_data(
        self,
        support_images
        support_labels
    ):
        """
        Extract support features, compute prototypes,
            and store support labels, features, and prototypes
        """
        self.support_labels = support_labels
        self.support_features = self.backbone(support_images)
        self.prototypes = compute_prototypes(self.support_features, support_labels)