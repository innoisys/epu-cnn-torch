import cv2 as cv
import numpy as np

from skimage.filters import sobel
from scipy.ndimage import gaussian_filter
from PIL import Image

class ImageToPFM(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self._output_size = output_size

    def __call__(self, image: Image):
        
        image = np.array(image)
        lab_rep = cv.cvtColor(image, cv.COLOR_RGB2LAB)
        l, a, b = cv.split(lab_rep)

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
