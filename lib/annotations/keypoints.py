import numpy as np
from scipy.stats import norm

class HeatMap(object):
    __instance = None
    def __new__(cls, dims):
        if HeatMap.__instance is None:
            HeatMap.__instance = object.__new__(cls)
        HeatMap.__instance.dims = dims
        return HeatMap.__instance

class Keypoints(object):
    def __init__(self, class_idx, bbox, coords, start_idx, kps_flips):
        self.__class_idx = class_idx
        self.bbox = bbox
        self.__start_idx = start_idx
        self.__kps_flips = kps_flips

        self.indexes = np.ma.any(coords, axis=1).nonzero()[0]
        self.__coords = np.ma.compress_rows(coords)

        HeatMap.sigma = 0.5
        HeatMap.thresh = 0.2

    def toline(self):
        normalized = self.normalized_coords()
        gaussian = self.gaussian_coords(normalized)

        kps_len = sum(map(lambda tup: len(tup[1]), gaussian))
        kps_start = self.__start_idx * HeatMap.dims[0] * HeatMap.dims[1]
        kps_end = kps_start + \
            len(self.__kps_flips) * HeatMap.dims[0] * HeatMap.dims[1] - 1
        kps_str = '{} {} {}'.format(kps_len, kps_start, kps_end)

        for kp_idx, kps in zip(self.indexes, gaussian):
            flip_kp_idx = self.__kps_flips[kp_idx]
            kp_idx += self.__start_idx
            flip_kp_idx += self.__start_idx

            for coords, value in zip(*kps):
                idx = kp_idx * HeatMap.dims[0] * HeatMap.dims[1] + \
                    coords[1] * HeatMap.dims[0] + coords[0]
                flip_idx = flip_kp_idx * HeatMap.dims[0] * HeatMap.dims[1] + \
                    coords[1] * HeatMap.dims[0] + \
                    (HeatMap.dims[0] - 1 - coords[0])
                kps_str += ' {} {} {}'.format(idx, flip_idx, value)

        # Object class should be 1-indexed
        return '{} {} {} {} {} {} {}\n'.format(
            self.__class_idx + 1, 1.0, *self.bbox, kps_str)

    def normalized_coords(self):
        x1, y1, x2, y2 = self.bbox
        dx = float(x2 - x1 + 1) / float(HeatMap.dims[0])
        dy = float(y2 - y1 + 1) / float(HeatMap.dims[1])

        coords = np.empty(self.__coords.shape, dtype=np.int)
        coords[:, 0] = np.floor((self.__coords[:, 0] - x1) / dx).astype(np.int)
        coords[:, 1] = np.floor((self.__coords[:, 1] - y1) / dy).astype(np.int)
        return coords

    @staticmethod
    def gaussian_coords(normalized):
        sigma = HeatMap.sigma

        gaussian = []
        for coords in normalized:
            pXs = norm.pdf(np.arange(HeatMap.dims[0]), coords[0], sigma) \
                / norm.pdf(0, 0, sigma)
            pYs = norm.pdf(np.arange(HeatMap.dims[1]), coords[1], sigma) \
                / norm.pdf(0, 0, sigma)
            pYs, pXs = np.meshgrid(pYs, pXs)
            pXYs = np.multiply(pXs, pYs)

            good_cond = pXYs >= HeatMap.thresh
            good_coords = np.argwhere(good_cond)
            good_values = pXYs[good_cond]
            gaussian.append((good_coords, good_values))

        return gaussian
