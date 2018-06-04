import collections
import numpy as np
import scipy.io as sio

from annotations.keypoints import HeatMap
from utils.bbox import is_bbox_valid

class Annotations(object):
    def __init__(self, classes, dataset, imgid, exclude_occluded=True):
        self.__exclude_occluded = exclude_occluded
        self.bboxes = collections.defaultdict(list)
        self.coords = collections.defaultdict(list)
        self.__total_kps = dataset.total_kps

        records_dict = dataset.records_by_imgid
        for idx, cls in enumerate(classes):
            data = sio.loadmat(dataset.matpath(cls, imgid))
            record = data['record'][0][0]

            if idx == 0:
                self.__imgpath = dataset.imgpath(cls, record['filename'][0])
                size = record['size'][0][0]
                self.width = size['width'][0][0]
                self.height = size['height'][0][0]
                self.depth = size['depth'][0][0]

            segkps = sio.loadmat(dataset.segkps_path(cls))
            keypoints = segkps['keypoints'][0][0]
            rec_indexes = records_dict[imgid][cls]

            # Make coordinates of keypoints to be 0-indexed
            coordinates = keypoints['coords'][rec_indexes] - 1
            obj_indexes = np.squeeze(
                keypoints['voc_rec_id'][rec_indexes], axis=1)
            # Make record indexes to be 0-based
            obj_indexes -= 1

            objects = record['objects'][0][obj_indexes]
            data = self.__read_class_data(cls, objects, coordinates)
            if data:
                bboxes, coordinates = zip(*data)
                self.bboxes[cls] = bboxes
                self.coords[cls] = coordinates

    def __read_class_data(self, cls, objects, coordinates):
        data = []
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
            if not is_bbox_valid(bbox, width=self.width, height=self.height):
                continue
            data.append((bbox, coords))

        return data

    def tolines(self):
        return '{}\n{}\n{}\n{}\n{}\n'.format(
            self.__imgpath, self.depth, self.height, self.width,
            self.__total_kps * HeatMap.dims[0] * HeatMap.dims[1])
