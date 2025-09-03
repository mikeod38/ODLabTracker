import numpy as np
from skimage import io, color, filters, morphology, measure

def preprocess_frame(img, min_area=800, max_area=2000):
    """
    Convert a frame to binary mask of worm-like objects.

    Parameters
    ----------
    img : ndarray
        Input frame (RGB or grayscale).
    min_area : int
        Minimum object area in pixels to keep.
    max_area : int
        Maximum object area in pixels to keep.

    Returns
    -------
    mask : ndarray (bool)
        Binary mask of worms only.
    props : list of regionprops
        Properties of detected objects.
    """
    # Convert to grayscale if RGB
    if img.ndim == 3:
        gray = color.rgb2gray(img)
    else:
        gray = img

    # Threshold (Otsu or adaptive)
    thresh = filters.threshold_otsu(gray)
    bw = gray < thresh   # worms usually darker

    # Remove small noise
    bw = morphology.remove_small_objects(bw, min_size=min_area)

    # Label objects
    labeled = measure.label(bw)
    props = measure.regionprops(labeled)

    # Filter by area range
    mask = np.zeros_like(bw, dtype=bool)
    for prop in props:
        if min_area <= prop.area <= max_area:
            mask[labeled == prop.label] = True

    return mask, props
