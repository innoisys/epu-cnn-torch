from typing import Tuple

import cv2 as cv
import numpy as np

from PIL import Image
from skimage.filters import sobel
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter


class ImageToPFM(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self._output_size = output_size

    def _rgb_to_lab(self, image: Image) -> Tuple[ArrayLike, ...]:
        """
        Convert a RGB image to LAB color space.
        """
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        try:
            image = np.asarray(image)
            lab_rep = cv.cvtColor(image, cv.COLOR_RGB2LAB)
            l, a, b = cv.split(lab_rep)
        except Exception as e:
            raise ValueError(f"Error converting image to PFM: {e}")
        
        return l, a, b

    def __call__(self, image: Image):
        
        l, a, b = self._rgb_to_lab(image)

        high_frequencies =  sobel(l)
        low_frequencies = gaussian_filter(l, sigma=3)

        pfm = [a, b, high_frequencies, low_frequencies]

        return pfm

class PFMToTensor(object):

    def __call__(self, pfm):
        from utils.epu_utils import min_max_normalization
        
        for i, _pfm in enumerate(pfm):
            _pfm = _pfm.astype(np.float32)
            _pfm = min_max_normalization(_pfm)
            pfm[i] = np.expand_dims(_pfm, axis=0)
        return pfm
