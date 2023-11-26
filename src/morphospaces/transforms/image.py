from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.stats import multivariate_normal
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


class Squeezed:
    """Remove singleton dimensions.

    For example, this would go from shape (1, Z, Y, X) -> (Z, Y, X).

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
            data_item.update({key: image.squeeze()})

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


class MaskFromVectorField:
    """Make a mask that is true where vector field magnitude is greater than 0.

    The vector field is assumed to be CZYX.
    The resulting mask has shape (z, y, x).

    Parameters
    ----------
    input_key : str
        The key for the vector field
    output_key : str
        The key to save the vector field to
    """

    def __init__(self, input_key: str, output_key: str):
        self.key = input_key
        self.output_key = output_key

    def __call__(self, data_item: Dict[str, np.ndarray]):
        vector_field = data_item[self.key]
        non_zero_components = vector_field != 0

        mask = np.sum(non_zero_components, axis=0) > 0

        data_item.update({self.output_key: np.expand_dims(mask, axis=0)})
        return data_item


class NormVectorField:
    """Compute the magnitude of a vector field.

    The vector field is assumed to be CZYX.
    The resulting image has shape (1, z, y, x).

    Parameters
    ----------
    input_key : str
        The key for the vector field
    output_key : str
        The key to save the image to.
    """

    def __init__(self, input_key: str, output_key: str):
        self.key = input_key
        self.output_key = output_key

    def __call__(self, data_item: Dict[str, np.ndarray]):
        vector_field = data_item[self.key]
        magnitude = np.linalg.norm(vector_field, axis=0)
        data_item.update({self.output_key: np.expand_dims(magnitude, axis=0)})
        return data_item


class RandPatchReduceIntensity:
    def __init__(self, label_key: str, image_key: str, patch_width: int):
        self.label_key = label_key
        self.image_key = image_key
        self.patch_width = patch_width

        # pre-compute the gaussian kernel
        (
            self.kernel_coordinates,
            self.kernel_values,
        ) = self.generate_3d_gaussian_kernel()

    def generate_3d_gaussian_kernel(self):
        """Generate a 3D Gaussian kernel."""
        # compute the grid coordinates centered
        patch_half_width = self.patch_width // 2
        z, y, x = np.meshgrid[
            -patch_half_width:patch_half_width:1,
            -patch_half_width:patch_half_width:1,
            -patch_half_width:patch_half_width:1,
        ]
        kernel_coordinates = np.column_stack([z.flat, y.flat, x.flat])

        # compute the width of the Gaussian kernel
        sigma_1d = self.patch_width / 6
        sigma = np.array([sigma_1d, sigma_1d, sigma_1d])
        covariance = np.diag(sigma**2)

        # get the values
        mu = np.array([0, 0, 0])
        kernel_values = multivariate_normal.pdf(
            kernel_coordinates, mean=mu, cov=covariance
        )
        max_normalized_kernel_values = kernel_values / np.max(kernel_values)

        return kernel_coordinates, max_normalized_kernel_values

    def create_gaussian_attenuation_patch(
        self,
        patch_centroid: np.ndarray,
        image_shape: Tuple[int, int, int],
        attenuation_factor: float = 0.5,
    ):
        # get the bounds of the image to clip the patch coordinates
        min_coordinates = np.array([0, 0, 0])
        max_coordinates = np.array(image_shape) - 1

        # shift the coordinates to be centered at the patch centroid
        shifted_coordinates = self.kernel_coordinates + patch_centroid

        # get a mask of coordinates inside the image bounds
        inside_bounds_mask = np.any(
            np.logical_and(
                shifted_coordinates >= min_coordinates,
                shifted_coordinates <= max_coordinates,
            ),
            axis=1,
        )

        # get the coordinates in the image bounds
        coordinates_in_image = shifted_coordinates[inside_bounds_mask]

        # get the kernel values int the image bounds
        kernel_values_in_image = (
            attenuation_factor * self.kernel_values[inside_bounds_mask]
        )

        return coordinates_in_image, kernel_values_in_image

    def __call__(
        self, data_item: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        # get the images
        image = data_item[self.image_key]
        label_image = data_item[self.label_key]

        label_mask = label_image != 0

        # get a random pixel coordinate in the foreground of the label mask
        pixel_coordinates = np.argwhere(label_mask)
        bounding_box_center = np.random.choice(pixel_coordinates, size=1)

        coordinates, kernel_values = self.create_gaussian_attenuation_patch(
            patch_centroid=bounding_box_center,
            image_shape=image.shape,
            attenuation_factor=0.5,
        )

        # make the attenuation map
        attenuation_map = np.ones_like(image)
        attenuation_map[tuple(coordinates.T)] = 1 - kernel_values

        # apply the attenuation map and store the result
        attenuated_image = image * attenuation_map
        data_item.update({self.image_key: attenuated_image})
        return data_item
