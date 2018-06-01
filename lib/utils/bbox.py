import numpy as np
from sklearn.utils.extmath import cartesian

def is_bbox_valid(bbox, width, height):
    x1, y1, x2, y2 = np.round(bbox).astype(np.int)
    return (x1 < width) and (y1 < height) and \
        (x2 >= 0) and (y2 >= 0) and (x2 >= x1) and (y2 >= y1)

def bbox_overlaps(bbox):
    dx = int(round((bbox[2] - bbox[0] + 1) / 6.0))
    dy = int(round((bbox[3] - bbox[1] + 1) / 6.0))
    shifts = cartesian(np.tile(np.array([-1, 0, 1]), (4, 1)))
    return np.apply_along_axis(lambda v: bbox + v * [dx, dy, dx, dy], 1, shifts)
