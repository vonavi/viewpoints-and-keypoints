import numpy as np
from sklearn.utils.extmath import cartesian

def is_bbox_valid(bbox, width, height):
    x1, y1, x2, y2 = bbox
    return (x1 < width) and (y1 < height) and \
        (x2 >= 0) and (y2 >= 0) and (x2 >= x1) and (y2 >= y1)

def bbox_overlaps(bbox):
    dx = float(bbox[2] - bbox[0] + 1) / float(6)
    dx = int(round(dx))
    dy = float(bbox[3] - bbox[1] + 1) / float(6)
    dy = int(round(dy))

    def shift_bbox(shift):
        return [bbox[0] + shift[0] * dx,
                bbox[1] + shift[1] * dy,
                bbox[2] + shift[2] * dx,
                bbox[3] + shift[3] * dy]

    shifts = cartesian(np.tile(np.array([-1, 0, 1]), (4, 1)))
    return np.apply_along_axis(shift_bbox, 1, shifts)
