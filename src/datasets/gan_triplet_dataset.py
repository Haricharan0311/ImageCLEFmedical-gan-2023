from pathlib import Path

from .easy_set import EasySet

DATA_DIR = ''

def GANTripletDataset(split, **kwargs):  # pylint: disable=invalid-name
    """
    Build the CUB dataset for the specific split.
    Args:
        split: one of the available split (typically train, val, test).
    Returns:
        the constructed dataset using EasySet
    """
    specs_file = CUB_SPECS_DIR / f"{split}.json"
    if specs_file.is_file():
        return EasySet(specs_file=specs_file, **kwargs)

    raise ValueError(f"Could not find specs file {specs_file.name} in {CUB_SPECS_DIR}")