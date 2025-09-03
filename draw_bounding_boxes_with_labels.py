import numpy as np
from skimage.draw import rectangle_perimeter
import matplotlib.pyplot as plt

def draw_bounding_boxes_with_labels(img, props, color=(255, 0, 0)):
    """
    Draw bounding boxes with labels on detected worms.

    Parameters
    ----------
    img : ndarray (RGB image)
    props : list of regionprops
    color : tuple
        RGB color for box outline.

    Returns
    -------
    overlay : ndarray
        Image with boxes drawn.
    """
    overlay = np.array(img).copy()

    for prop in props:
        minr, minc, maxr, maxc = prop.bbox

        # Draw box
        rr, cc = rectangle_perimeter(
            (minr, minc), end=(maxr, maxc), shape=overlay.shape[:2]
        )
        overlay[rr, cc] = color

    # Show with labels
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(overlay)
    ax.axis("off")

    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        ax.text(
            minc, minr - 5, f"{prop.label}",
            color="yellow", fontsize=12, weight="bold"
        )

    plt.show()

    return overlay
