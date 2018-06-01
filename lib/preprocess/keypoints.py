import numpy as np
from scipy.stats import norm

HEAT_MAP_DIMS = [6, 6]
HEAT_MAP_THRESH = 0.2

class Keypoints(object):
    def __init__(self, class_idx, bbox, coords, start_idx, kps_flips):
        self.__class_idx = class_idx
        self.__bbox = bbox
        self.__start_idx = start_idx
        self.__kps_flips = kps_flips

        good_coords = np.all(~np.isnan(coords), axis=1)
        self.__indexes = good_coords.nonzero()[0]
        self.__coords = coords[good_coords]

    def toline(self):
        normalized = self.normalized_coords()
        gaussian = self.gaussian_coords(normalized, HEAT_MAP_THRESH)

        kps_len = sum(map(lambda tup: len(tup[1]), gaussian))
        kps_start = self.__start_idx * HEAT_MAP_DIMS[0] * HEAT_MAP_DIMS[1]
        kps_end = kps_start + \
            len(self.__kps_flips) * HEAT_MAP_DIMS[0] * HEAT_MAP_DIMS[1] - 1
        kps_str = '{} {} {}'.format(kps_len, kps_start, kps_end)

        for kp_idx, kps in zip(self.__indexes, gaussian):
            flip_kp_idx = self.__kps_flips[kp_idx]
            kp_idx += self.__start_idx
            flip_kp_idx += self.__start_idx

            for coords, value in zip(*kps):
                idx = kp_idx * HEAT_MAP_DIMS[0] * HEAT_MAP_DIMS[1] + \
                    coords[1] * HEAT_MAP_DIMS[0] + coords[0]
                flip_idx = flip_kp_idx * HEAT_MAP_DIMS[0] * HEAT_MAP_DIMS[1] + \
                    coords[1] * HEAT_MAP_DIMS[0] + \
                    (HEAT_MAP_DIMS[0] - 1 - coords[0])
                kps_str += ' {} {} {}'.format(idx, flip_idx, value)

        # Object class should be 1-indexed
        return '{} {} {} {} {} {} {}\n'.format(
            self.__class_idx + 1, 1.0, *self.__bbox, kps_str)

    def normalized_coords(self):
        x1, y1, x2, y2 = self.__bbox
        dx = float(x2 - x1 + 1) / float(HEAT_MAP_DIMS[0])
        dy = float(y2 - y1 + 1) / float(HEAT_MAP_DIMS[1])

        coords = np.empty(self.__coords.shape, dtype=np.int)
        coords[:, 0] = np.floor((self.__coords[:, 0] - x1) / dx).astype(np.int)
        coords[:, 1] = np.floor((self.__coords[:, 1] - y1) / dy).astype(np.int)
        return coords

    @staticmethod
    def gaussian_coords(normalized, prob_thresh):
        sigma = 0.5

        gaussian = []
        for coords in normalized:
            pXs = norm.pdf(np.arange(HEAT_MAP_DIMS[0]), coords[0], sigma) \
                / norm.pdf(0, 0, sigma)
            pYs = norm.pdf(np.arange(HEAT_MAP_DIMS[1]), coords[1], sigma) \
                / norm.pdf(0, 0, sigma)
            pYs, pXs = np.meshgrid(pYs, pXs)
            pXYs = np.multiply(pXs, pYs)

            good_cond = pXYs >= prob_thresh
            good_coords = np.argwhere(good_cond)
            good_values = pXYs[good_cond]
            gaussian.append((good_coords, good_values))

        return gaussian
