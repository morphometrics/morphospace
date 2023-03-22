from typing import Dict, List, Union

import numpy as np
from skimage.util import img_as_float32


class ExpandDimsd:
    """Prepend a singleton dimensions.

    For example, this would go from shape (Z, Y, X) -> (1, Z, Y, X).

    This is intended to be used with datasets where the
    data are loaded as a dictionary.

    Parameters
    ----------
    keys : Union[str, List[str]]
        The keys in the dataset to apply the transform to.
    """

    def __init__(self, keys: Union[str, List[str]]):
        if isinstance(keys, str):
            keys = [keys]
        self.keys: List[str] = keys

    def __call__(
        self, data_item: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        for key in self.keys:
            image = data_item[key]
            data_item.update({key: np.expand_dims(image, axis=0)})

        return data_item


class ImageAsFloat32:
    """Convert an image to a float32 ranging from 0 to 1.

    Parameters
    ----------
    keys : Union[str, List[str]]
        The keys in the dataset to apply the transform to.
    """

    def __init__(self, keys: Union[str, List[str]]):
        if isinstance(keys, str):
            keys = [keys]
        self.keys: List[str] = keys

    def __call__(
        self, data_item: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        for key in self.keys:
            image = data_item[key]
            data_item.update({key: img_as_float32(image)})

        return data_item
