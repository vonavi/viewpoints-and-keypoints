import collections
import numpy as np
import scipy.io as sio
from scipy.stats import norm
from sklearn.utils.extmath import cartesian

HEAT_MAP_DIMS = [6, 6]
HEAT_MAP_THRESH = 0.2

class Keypoints(object):
    def __init__(self, class_idx, bbox, coords, start_idx, kps_flips):
        self.__class_idx = class_idx
        self.__bbox = bbox
        self.__start_idx = start_idx
        self.__kps_flips = kps_flips

        all_indexes = np.arange(coords.shape[0])
        bad_indexes = np.where(np.isnan(coords))[0]
        good_indexes = np.setdiff1d(all_indexes, bad_indexes)
        self.__indexes = good_indexes
        self.__coords = coords[good_indexes]

    def toline(self):
        normalized = self.normalized_coords()
        gaussian = self.gaussian_coords(normalized, HEAT_MAP_THRESH)

        kps_str = ''
        count = 0
        for kp_idx, kps in zip(self.__indexes, gaussian):
            if kps[1].size == 0:
                continue

            flip_kp_idx = self.__kps_flips[kp_idx]
            kp_idx += self.__start_idx
            flip_kp_idx += self.__start_idx
            for coords, value in zip(*kps):
                count += 1
                idx = kp_idx * HEAT_MAP_DIMS[0] * HEAT_MAP_DIMS[1] + \
                    coords[1] * HEAT_MAP_DIMS[0] + coords[0]
                flip_idx = flip_kp_idx * HEAT_MAP_DIMS[0] * HEAT_MAP_DIMS[1] + \
                    coords[1] * HEAT_MAP_DIMS[0] + \
                    (HEAT_MAP_DIMS[0] - 1 - coords[0])
                kps_str += ' {} {} {}'.format(idx, flip_idx, value)

        kps_start = self.__start_idx * HEAT_MAP_DIMS[0] * HEAT_MAP_DIMS[1]
        kps_end = kps_start + \
            len(self.__kps_flips) * HEAT_MAP_DIMS[0] * HEAT_MAP_DIMS[1] - 1
        kps_str = '{} {} {}'.format(count, kps_start, kps_end) + kps_str

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

class Annotations(object):
    def __init__(self, classes, dataset, imgid, exclude_occluded=True):
        self.__exclude_occluded = exclude_occluded
        self.__data = []
        self.__parts = self.get_parts_from(dataset)
        start_indexes, self.__total_kps = self.get_start_indexes(dataset)
        kps_flips = self.keypoint_flips(self.__parts)

        for idx, cls in enumerate(classes):
            data = sio.loadmat(dataset.matpath(cls, imgid))
            record = data['record'][0][0]

            if idx == 0:
                self.__imgpath = dataset.imgpath(cls, record['filename'][0])
                size = record['size'][0][0]
                self.__width = size['width'][0][0]
                self.__height = size['height'][0][0]
                self.__depth = size['depth'][0][0]

            segkps = sio.loadmat(dataset.segkps_path(cls))
            keypoints = segkps['keypoints'][0][0]
            imgid_dict = self.records_by_imgid(keypoints['voc_image_id'])
            rec_indexes = imgid_dict[imgid]

            # Make coordinates of keypoints to be 0-indexed
            coordinates = keypoints['coords'][rec_indexes] - 1
            obj_indexes = np.squeeze(
                keypoints['voc_rec_id'][rec_indexes], axis=1)
            # Make record indexes to be 0-based
            obj_indexes = obj_indexes - 1

            class_idx = dataset.CLASSES.index(cls)
            objects = record['objects'][0][obj_indexes]
            self.read_class_data(
                cls, class_idx, objects, coordinates,
                start_idx=start_indexes[class_idx],
                kps_flips=kps_flips[class_idx])

    @staticmethod
    def get_parts_from(dataset):
        parts = []
        for cls in dataset.CLASSES:
            segkps = sio.loadmat(dataset.segkps_path(cls))
            keypoints = segkps['keypoints'][0][0]

            def gen_labels():
                for row in keypoints['labels']:
                    for label in row[0]:
                        yield label
            parts.append(np.stack(gen_labels()))

        return parts

    def get_start_indexes(self, dataset):
        indexes = np.zeros(len(self.__parts), dtype=np.int)
        start_idx = 0
        for idx in dataset.ANNOTATED:
            indexes[idx] = start_idx
            start_idx += len(self.__parts[idx])
        return (indexes, start_idx)

    @staticmethod
    def keypoint_flips(parts):
        kps_flips = []
        str_flips = {'L_': 'R_', 'Left': 'Right', 'left': 'right'}

        for class_parts in parts:
            flip_dict = dict()
            for idx, part_name in enumerate(class_parts):
                left_to_right = part_name
                for str_left, str_right in str_flips.items():
                    left_to_right = left_to_right.replace(str_left, str_right)

                right_to_left = part_name
                for str_left, str_right in str_flips.items():
                    right_to_left = right_to_left.replace(str_right, str_left)

                if left_to_right != part_name:
                    flip_dict[left_to_right] = idx
                elif right_to_left != part_name:
                    flip_dict[right_to_left] = idx
                else:
                    flip_dict[part_name] = idx

            flips = np.array([flip_dict[name] for name in class_parts])
            kps_flips.append(flips)

        return kps_flips

    @staticmethod
    def records_by_imgid(imgid_mat):
        imgid_dict = collections.defaultdict(list)
        for idx, row in enumerate(imgid_mat):
            imgid = row[0][0]
            imgid_dict[imgid].append(idx)
        return imgid_dict

    def read_class_data(
            self, cls, class_idx, objects, coordinates, start_idx, kps_flips):

        for obj, coords in zip(objects, coordinates):
            obj_class = obj['class'][0]
            difficult = bool(obj['difficult'][0][0])
            if obj_class != cls or difficult:
                continue

            if self.__exclude_occluded:
                occluded = bool(obj['occluded'][0][0])
                truncated = bool(obj['truncated'][0][0])
                if occluded or truncated:
                    continue

            # Convert the bounding box from 1- to 0-indexed
            bbox = obj['bbox'][0] - 1
            bbox = np.round(bbox).astype(np.int)
            if not self.is_bbox_valid(bbox):
                continue

            for box in self.overlapping_boxes(bbox):
                keypoints = Keypoints(
                    class_idx=class_idx, bbox=box, coords=coords,
                    start_idx=start_idx, kps_flips=kps_flips)
                self.__data.append(keypoints)

    def is_bbox_valid(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 < self.__width) and (y1 < self.__height) and \
            (x2 >= 0) and (y2 >= 0) and (x2 >= x1) and (y2 >= y1)

    @staticmethod
    def overlapping_boxes(bbox):
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

    def is_empty(self):
        return len(self.__data) == 0

    def tolines(self):
        lines = '{}\n{}\n{}\n{}\n{}\n'.format(
            self.__imgpath, self.__depth, self.__height, self.__width,
            self.__total_kps * HEAT_MAP_DIMS[0] * HEAT_MAP_DIMS[1])
        lines += '{}\n'.format(len(self.__data))
        lines += ''.join(map(lambda x: x.toline(), self.__data))
        return lines
