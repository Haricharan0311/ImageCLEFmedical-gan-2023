import torch
from torchvision import io
import os
import itertools
import glob

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
        self.original_l = []
        self.generated_l = []
        self._generate_pairs()   # sets self.pairs_l
        
        print(f"Initialized '{mode}' data from {self.img_concrete_path}")

    
    def __len__(self):
        # return 4
        return len(self.pairs_l)


    def __getitem__(self, idx):
        """
        - Generates a single sample --> (real_img, generate_img, ideal_sim_score).
        - Randomly, but exhaustively, generate pairs from the triplet.
        """
        
        if self.data_mode == 'test':
            img_path_1, img_path_2 = self.pairs_l[idx]
            try:
                real_img = io.read_image(img_path_1).float()
                generated_img = io.read_image(img_path_2).float()
            except FileNotFoundError as e:
                print("Error when trying to read data file:", e)
                return None            
            return (real_img, generated_img)
        
        else:
            img_path_1, img_path_2, similarity_scores = self.pairs_l[idx]
            try:
                real_img = io.read_image(img_path_1).float()
                generated_img = io.read_image(img_path_2).float()
            except FileNotFoundError as e:
                print("Error when trying to read data file:", e)
                return None            
            return (real_img, generated_img, float(similarity_scores))


    def get_test_size(self):
        return len(self.original_l)

    
    def get_test_samples(self, idx):
        """ 
        Generator to yield all 'generated' samples for each 'original' image.
        To cumulatively assess the class of an 'original' image.
        """

        print("[INFO] Do NOT set up the 'dataloader' for random sampling with this generator!")

        def supply_all_generated(orig_path, is_real):
            
            orig_img = io.read_image(orig_path).float().unsqueeze(dim=0)
            for gen_path in self.generated_l:
                try:
                    generated_img = io.read_image(gen_path).float().unsqueeze(dim=0)
                    is_real = float(is_real) if is_real is not None else is_real
                    yield orig_img, generated_img, torch.Tensor([is_real])
                except FileNotFoundError as e:
                    print("Error when trying to read data file:", e)
                    # TODO: Will break anyway. Traceforward more suitably.
                    yield None, None, None  

        return supply_all_generated(*self.original_l[idx])


    def _generate_pairs_from_triplets(self):

        def prefix_source_path(img_fname):
            return os.path.join(self.img_concrete_path, img_fname)

        real_used = glob.glob(os.path.join(self.img_concrete_path, 'real_used/*.png'))
        real_unused = glob.glob(os.path.join(self.img_concrete_path, 'real_unused/*.png'))
        generated = glob.glob(os.path.join(self.img_concrete_path, 'generated/*.png'))

        all_pairs = []
        all_pairs.extend(list(itertools.product(*[real_used, generated, [1]])))
        all_pairs.extend(list(itertools.product(*[real_unused, generated, [0]])))
        self.pairs_l = list(all_pairs)
        
        original_l = []
        original_l.extend([(img_path, 1) for img_path in real_used])
        original_l.extend([(img_path, 0) for img_path in real_unused])
        self.original_l = list(original_l)

        self.generated_l = list(generated)


    def _generate_pairs_from_couplets(self):
        
        def prefix_source_path(img_fname):
            return os.path.join(self.img_concrete_path, img_fname)

        real = glob.glob(os.path.join(self.img_concrete_path, 'real_all/*.png'))
        generated = glob.glob(os.path.join(self.img_concrete_path, 'generated/*.png'))

        self.pairs_l = list(itertools.product(*[real, generated]))        
        self.original_l = [(img_path, None) for img_path in real]
        self.generated_l = list(generated)


    def _generate_pairs(self):
        if self.data_mode in ['train', 'validate']:
            self._generate_pairs_from_triplets()
        else:
            self._generate_pairs_from_couplets()


# test_obj = GANTripletDataset(mode='train')

# from torch.utils.data import DataLoader 
# test_dloader = DataLoader(
#     test_obj,
#     batch_size=4,
#     pin_memory=True,
#     shuffle=True,
# )

# for x in test_dloader:
#     print(x)