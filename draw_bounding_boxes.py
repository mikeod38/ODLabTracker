import numpy as np
from skimage.draw import rectangle_perimeter
import matplotlib.pyplot as plt

def draw_bounding_boxes(img, props, color=(255, 0, 0)):
    """
    Draw red bounding boxes on detected worms.

    img : ndarray (RGB image)
    props : list of regionprops
    color : tuple, RGB color for box
    """
    overlay = np.array(img).copy()
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        rr, cc = rectangle_perimeter(
            (minr, minc), end=(maxr, maxc), shape=overlay.shape[:2]
        )
        overlay[rr, cc] = color
    return overlay

# Example use:
overlay_img = draw_bounding_boxes(img, props)
plt.imshow(overlay_img)
plt.axis("off")
plt.show()
