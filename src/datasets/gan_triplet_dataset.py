from pathlib import Path
import os
import itertools

DATA_DIR = os.path.join(os.path.split(__file__)[0], '../../data/gan_triplet')


class GANTripletDataset:  # pylint: disable=invalid-name
    """
    Build the dataset for the specific split.
    """

    def __init__(self, root=DATA_DIR, mode='train', transform=None, target_transform=None):

        """ 
        - mode: one of `train`, `validate`, or `test` 
        """

        # Ensure data path validity
        self.img_base_path = root
        assert os.path.isdir(self.img_base_path), f"Need valid data path as `root`. Got {root}"
        self.img_concrete_path = os.path.join(
            self.img_base_path,
            mode 
        )
        assert os.path.isdir(self.img_concrete_path), f"Could not find valid data path at {self.img_concrete_path}"
        
        self.data_mode = mode

        self.pairs_l = []
        self._generate_pairs()   # sets self.pairs_l
        
        print(f"Initialized '{mode}' data from {self.img_concrete_path}")

    
    def __len__(self):
        return len(self.pairs_l)


    def __getitem__(self, idx):
        """
        - Generates a single sample --> (real_img, generate_img, ideal_sim_score).
        - Randomly, but exhaustively, generate pairs from the triplet.
        """
        pass 


    def _generate_pairs_from_triplets(self):

        def prefix_source_path(img_fname):
            return os.path.join(self.img_concrete_path, img_fname)

        real_used = list(map(prefix_source_path, os.listdir(os.path.join(self.img_concrete_path, 'real_used'))))
        real_unused = list(map(prefix_source_path, os.listdir(os.path.join(self.img_concrete_path, 'real_unused'))))
        generated = list(map(prefix_source_path, os.listdir(os.path.join(self.img_concrete_path, 'generated'))))

        all_pairs = []
        all_pairs.extend(list(itertools.product(*[real_used, generated, [1]])))
        all_pairs.extend(list(itertools.product(*[real_unused, generated, [0]])))
        self.pairs_l = list(all_pairs)


    def _generate_pairs(self):
        if self.data_mode in ['train', 'validate']:
            self._generate_pairs_from_triplets()
        else:
            self._generate_pairs_from_couplets()


test_obj = GANTripletDataset()