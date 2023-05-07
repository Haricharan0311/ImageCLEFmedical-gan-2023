import torch
from torch import Tensor, nn

from .blocks import default_relation_module
from utils import compute_prototypes

from .comparative_model_base import ComparativeModelBase


class RelationNetwork(ComparativeModelBase):
    """
    - extract feature maps for the image to be compared.
	- concatenate feature maps of the two images.
	- feed into a relation module, i.e. a CNN that outputs a relation score.     
	- NOTE: Feature extraction here is multi-dimensional -- don't flatten in the final layer before relation module.
    """

    def __init__(self, *args, relation_module=None, **kwargs):
        """
        Build Relation Network by calling the constructor of ComparativeModelBase.
        """
        super().__init__(*args, **kwargs)

        if len(self.backbone_output_shape) != 3:
            raise ValueError(
                "Illegal backbone for Relation Networks. Expected output for an image is a 3-dim "
                "tensor of shape (n_channels, width, height)."
            )

        self.relation_module = (
            relation_module
            if relation_module
            else default_relation_module(self.feature_dimension)
        )

    def process_support_set(
        self,
        support_images,
        support_labels,
    ):
        """
        [FOR LATER USE]
        Overrides process_support_set of ComparativeModelBase.
        Extract feature maps from the support set and store class prototypes.
        """

        support_features = self.backbone(support_images)
        self.prototypes = compute_prototypes(support_features, support_labels)

    def forward(self, input_images):
        """
        Overrides method forward in ComparativeModelBase.
        - Concatenate feature maps from two images
        - Feed the result into a relation module.
        """

        real_image, gen_image = input_images
        real_img_feats = self.backbone(real_image)
        gen_img_feats = self.backbone(gen_image)

        # For each pair (query, prototype), we compute the concatenation of their feature maps
        # Given that query_features is of shape (n_queries, n_channels, width, height), the
        # constructed tensor is of shape (n_queries * n_prototypes, 2 * n_channels, width, height)
        # (2 * n_channels because prototypes and queries are concatenated)
        query_prototype_feature_pairs = torch.cat(
            (
                self.prototypes.unsqueeze(dim=0).expand(
                    query_features.shape[0], -1, -1, -1, -1
                ),
                query_features.unsqueeze(dim=1).expand(
                    -1, self.prototypes.shape[0], -1, -1, -1
                ),
            ),
            dim=2,
        ).view(-1, 2 * self.feature_dimension, *query_features.shape[2:])

        # Each pair (query, prototype) is assigned a relation scores in [0,1]. Then we reshape the
        # tensor so that relation_scores is of shape (n_queries, n_prototypes).
        relation_scores = self.relation_module(query_prototype_feature_pairs).view(
            -1, self.prototypes.shape[0]
        )

        return self.softmax_if_specified(relation_scores)

    @staticmethod
    def is_transductive():
        return False