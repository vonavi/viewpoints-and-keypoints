import errno
import os

import math
import numpy as np
import scipy.io as sio

CLASSES = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair',
           'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']

class Pascal(object):
    def __init__(self, root):
        self.__name = 'pascal'
        self.__root = os.path.normpath(root)

    def load_record(self, cls, imgid):
        matpath = os.path.join(
            self.__root, 'Annotations', cls + '_' + self.__name, imgid + '.mat')
        data = sio.loadmat(matpath)
        return data['record'][0][0]

    def get_imgpath(self, cls, record):
        imgname = record['imgname'][0]
        imgpath = os.path.join(self.__root, 'PASCAL', 'VOCdevkit', imgname)
        if not os.path.isfile(imgpath):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), imgpath)
        return imgpath

    def read_class_set(self, cls, phase):
        setname = cls + '_' + phase + '.txt'
        setpath = os.path.join(
            self.__root, 'PASCAL', 'VOCdevkit', 'VOC2012',
            'ImageSets', 'Main', setname)

        def gen_set(lines):
            for line in lines:
                words = line.split()
                if words[1] == '-1':
                    continue

                item = np.array(
                    [(cls, self, words[0])], dtype=[
                        ('class', 'U16'), ('dataset', 'object'), ('imgid', 'U16')])
                yield item

        with open(setpath, 'r') as f:
            lines = f.read().splitlines()
            imgset = np.stack(gen_set(lines))

        return imgset

class Imagenet(object):
    def __init__(self, root):
        self.__name = 'imagenet'
        self.__root = os.path.normpath(root)

    def load_record(self, cls, imgid):
        matpath = os.path.join(
            self.__root, 'Annotations', cls + '_' + self.__name, imgid + '.mat')
        data = sio.loadmat(matpath)
        return data['record'][0][0]

    def get_imgpath(self, cls, record):
        filename = record['filename'][0]
        imgpath = os.path.join(
            self.__root, 'Images', cls + '_' + self.__name, filename)
        if not os.path.isfile(imgpath):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), imgpath)
        return imgpath

    def read_class_set(self, cls, phase):
        setname = '_'.join([cls, self.__name, phase]) + '.txt'
        setpath = os.path.join(self.__root, 'Image_sets', setname)

        def gen_set(lines):
            for line in lines:
                item = np.array(
                    [(cls, self, line)], dtype=[
                        ('class', 'U16'), ('dataset', 'object'), ('imgid', 'U16')])
                yield item

        with open(setpath, 'r') as f:
            lines = f.read().splitlines()
            imgset = np.stack(gen_set(lines))

        return imgset

class Pose(object):
    def __init__(self, cls, bbox, azimuth, elevation, theta):
        self.cls = cls
        self.bbox = bbox
        self.azimuth = azimuth
        self.elevation = elevation
        self.theta = theta

    def toline(self):
        # Object class should be 1-indexed
        return '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
            CLASSES.index(self.cls) + 1, 1.0, *self.bbox,
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
    def __init__(self, cls, dataset, imgid, exclude_occluded=False):
        record = dataset.load_record(cls, imgid)
        self.__imgpath = dataset.get_imgpath(cls, record)

        size = record['size'][0][0]
        self.__width = size['width'][0][0]
        self.__height = size['height'][0][0]
        self.__depth = size['depth'][0][0]

        objects = record['objects'][0]
        self.__data = self.read_data(objects, cls, exclude_occluded)

    def read_data(self, objects, cls, exclude_occluded):
        data = []

        for obj in list(objects):
            obj_class = obj['class'][0]
            difficult = bool(obj['difficult'][0][0])
            if obj_class != cls or difficult:
                continue

            if exclude_occluded:
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
            boxes = self.overlapping_boxes(bbox)
            def create_pose(box):
                return Pose(
                    cls=cls, bbox=box, azimuth=azimuth, elevation=elevation,
                    theta=theta)
            poses = np.apply_along_axis(create_pose, 1, boxes)
            data.append(poses)

        return data

    @staticmethod
    def overlapping_boxes(bbox):
        dx = float(bbox[2] - bbox[0]) / float(6)
        dy = float(bbox[3] - bbox[1]) / float(6)

        def gen_boxes():
            for x1_shift in -1, 0, 1:
                for y1_shift in -1, 0, 1:
                    for x2_shift in -1, 0, 1:
                        for y2_shift in -1, 0, 1:
                            yield [bbox[0] + x1_shift * dx,
                                   bbox[1] + y1_shift * dy,
                                   bbox[2] + x2_shift * dx,
                                   bbox[3] + y2_shift * dy]

        boxes = np.stack(gen_boxes())
        boxes = np.round(boxes).astype(np.int)
        return boxes

    def tolines(self):
        lines = '{}\n{}\n{}\n{}\n'.format(
            self.__imgpath, self.__depth, self.__height, self.__width)

        if self.__data:
            poses = np.concatenate(self.__data)
            lines += '{}\n'.format(poses.size)
            lines += ''.join(map(lambda x: x.toline(), poses))
        else:
            lines += '0\n'

        return lines
