import math
import numpy as np
import scipy.io as sio
from sklearn.utils.extmath import cartesian

class Pose(object):
    def __init__(self, class_idx, bbox, azimuth, elevation, theta):
        self.class_idx = class_idx
        self.bbox = bbox
        self.azimuth = azimuth
        self.elevation = elevation
        self.theta = theta

    def toline(self):
        # Object class should be 1-indexed
        return '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
            self.class_idx + 1, 1.0, *self.bbox,
            math.ceil(self.theta * 10.5/180 + 9.5),
            math.ceil(- self.theta * 10.5/180 + 9.5),
            math.ceil(self.elevation * 10.5/180 + 9.5),
            math.ceil(self.elevation * 10.5/180 + 9.5),
            math.floor(self.azimuth * 10.5/180),
            20 - math.floor(self.azimuth * 10.5/180),
            math.ceil(self.theta * 3.5/180 + 2.5),
            math.ceil(- self.theta * 3.5/180 + 2.5),
            math.ceil(self.elevation * 3.5/180 + 2.5),
            math.ceil(self.elevation * 3.5/180 + 2.5),
            math.floor(self.azimuth * 3.5/180),
            6 - math.floor(self.azimuth * 3.5/180))

class Annotations(object):
    def __init__(self, classes, dataset, imgid, exclude_occluded=False):
        self.__exclude_occluded = exclude_occluded
        self.__data = []

        for idx, cls in enumerate(classes):
            data = sio.loadmat(dataset.matpath(cls, imgid))
            record = data['record'][0][0]

            if idx == 0:
                self.__imgpath = dataset.imgpath(cls, record['filename'][0])
                size = record['size'][0][0]
                self.__width = size['width'][0][0]
                self.__height = size['height'][0][0]
                self.__depth = size['depth'][0][0]

            class_idx = dataset.CLASSES.index(cls)
            objects = record['objects'][0]
            self.read_class_data(cls, class_idx, objects)

    def read_class_data(self, cls, class_idx, objects):
        for obj in list(objects):
            obj_class = obj['class'][0]
            difficult = bool(obj['difficult'][0][0])
            if obj_class != cls or difficult:
                continue

            if self.__exclude_occluded:
                occluded = bool(obj['occluded'][0][0])
                truncated = bool(obj['truncated'][0][0])
                if occluded or truncated:
                    continue

            viewpoint = obj['viewpoint'][0][0]
            azimuth = viewpoint['azimuth'][0][0]
            elevation = viewpoint['elevation'][0][0]
            theta = viewpoint['theta'][0][0]

            # Convert the bounding box from 1- to 0-indexed
            bbox = obj['bbox'][0] - 1
            bbox = np.round(bbox).astype(np.int)
            if not self.is_bbox_valid(bbox):
                continue

            for box in self.overlapping_boxes(bbox):
                pose = Pose(
                    class_idx=class_idx, bbox=box, azimuth=azimuth,
                    elevation=elevation, theta=theta)
                self.__data.append(pose)

    def is_bbox_valid(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 < self.__width) and (y1 < self.__height) and \
            (x2 >= 0) and (y2 >= 0) and (x2 >= x1) and (y2 >= y1)

    def overlapping_boxes(self, bbox):
        dx = float(bbox[2] - bbox[0] + 1) / float(6)
        dx = int(round(dx))
        dy = float(bbox[3] - bbox[1] + 1) / float(6)
        dy = int(round(dy))

        def shift_bbox(shift):
            x1 = max(bbox[0] + shift[0] * dx, 0)
            y1 = max(bbox[1] + shift[1] * dy, 0)
            x2 = min(bbox[2] + shift[2] * dx, self.__width - 1)
            y2 = min(bbox[3] + shift[3] * dy, self.__height - 1)
            return [x1, y1, x2, y2]

        shifts = cartesian(np.tile(np.array([-1, 0, 1]), (4, 1)))
        boxes = np.apply_along_axis(shift_bbox, 1, shifts)
        boxes = np.unique(boxes, axis=0)
        return boxes

    def is_empty(self):
        return len(self.__data) == 0

    def tolines(self):
        lines = '{}\n{}\n{}\n{}\n'.format(
            self.__imgpath, self.__depth, self.__height, self.__width)
        lines += '{}\n'.format(len(self.__data))
        lines += ''.join(map(lambda x: x.toline(), self.__data))
        return lines
