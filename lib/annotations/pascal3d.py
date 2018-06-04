import math
import collections
import numpy as np
import scipy.io as sio

from utils.bbox import is_bbox_valid

class Pose(object):
    def __init__(self, class_idx, bbox, azimuth, elevation, theta):
        self.__class_idx = class_idx
        self.__bbox = bbox
        self.__azimuth = azimuth
        self.__elevation = elevation
        self.__theta = theta

    def toline(self):
        # Object class should be 1-indexed
        return '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
            self.__class_idx + 1, 1.0, *self.__bbox,
            math.ceil(self.__theta * 10.5/180 + 9.5),
            math.ceil(- self.__theta * 10.5/180 + 9.5),
            math.ceil(self.__elevation * 10.5/180 + 9.5),
            math.ceil(self.__elevation * 10.5/180 + 9.5),
            math.floor(self.__azimuth * 10.5/180),
            20 - math.floor(self.__azimuth * 10.5/180),
            math.ceil(self.__theta * 3.5/180 + 2.5),
            math.ceil(- self.__theta * 3.5/180 + 2.5),
            math.ceil(self.__elevation * 3.5/180 + 2.5),
            math.ceil(self.__elevation * 3.5/180 + 2.5),
            math.floor(self.__azimuth * 3.5/180),
            6 - math.floor(self.__azimuth * 3.5/180))

class Annotations(object):
    def __init__(self, classes, dataset, imgid, exclude_occluded=True):
        self.__exclude_occluded = exclude_occluded
        self.data = collections.defaultdict(list)

        for idx, cls in enumerate(classes):
            data = sio.loadmat(dataset.matpath(cls, imgid))
            record = data['record'][0][0]

            if idx == 0:
                self.__imgpath = dataset.imgpath(cls, record['filename'][0])
                size = record['size'][0][0]
                self.width = size['width'][0][0]
                self.height = size['height'][0][0]
                self.depth = size['depth'][0][0]

            objects = record['objects'][0]
            self.data[cls] = self.__read_class_data(cls, objects)

    def __read_class_data(self, cls, objects):
        data = []
        for obj in objects:
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
            if not is_bbox_valid(bbox, width=self.width, height=self.height):
                continue

            viewpoint = obj['viewpoint'][0][0]
            azimuth = viewpoint['azimuth'][0][0]
            elevation = viewpoint['elevation'][0][0]
            theta = viewpoint['theta'][0][0]
            data.append({'bbox': bbox,
                         'azimuth': azimuth,
                         'elevation': elevation,
                         'theta': theta})

        return data

    def tolines(self):
        return '{}\n{}\n{}\n{}\n'.format(
            self.__imgpath, self.depth, self.height, self.width)
