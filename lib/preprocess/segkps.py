import collections
import numpy as np
import scipy.io as sio

from preprocess.keypoints import HEAT_MAP_DIMS, Keypoints
from utils.bbox import *

class Annotations(object):
    def __init__(self, classes, dataset, imgid, exclude_occluded=True):
        self.__exclude_occluded = exclude_occluded
        self.__data = []
        self.__total_kps = dataset.total_kps

        records_dict = dataset.records_by_imgid
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
            rec_indexes = records_dict[imgid][cls]

            # Make coordinates of keypoints to be 0-indexed
            coordinates = keypoints['coords'][rec_indexes] - 1
            obj_indexes = np.squeeze(
                keypoints['voc_rec_id'][rec_indexes], axis=1)
            # Make record indexes to be 0-based
            obj_indexes -= 1

            class_idx = dataset.CLASSES.index(cls)
            objects = record['objects'][0][obj_indexes]
            self.read_class_data(
                cls, class_idx, objects, coordinates,
                start_idx=dataset.start_indexes[class_idx],
                kps_flips=dataset.kps_flips[class_idx])

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
            if not is_bbox_valid(bbox, width=self.__width, height=self.__height):
                continue

            for box in bbox_overlaps(bbox):
                keypoints = Keypoints(
                    class_idx=class_idx, bbox=box, coords=coords,
                    start_idx=start_idx, kps_flips=kps_flips)
                self.__data.append(keypoints)

    def is_empty(self):
        return len(self.__data) == 0

    def tolines(self):
        lines = '{}\n{}\n{}\n{}\n{}\n'.format(
            self.__imgpath, self.__depth, self.__height, self.__width,
            self.__total_kps * HEAT_MAP_DIMS[0] * HEAT_MAP_DIMS[1])
        lines += '{}\n'.format(len(self.__data))
        lines += ''.join(map(lambda x: x.toline(), self.__data))
        return lines
