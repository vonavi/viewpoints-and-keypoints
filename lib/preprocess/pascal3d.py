import os

import math
import numpy as np
import scipy.io as sio

CLASSES = {'aeroplane': 1, 'bicycle': 2, 'boat': 4, 'bottle': 5, 'bus': 6,
           'car': 7, 'chair': 9, 'diningtable': 11, 'motorbike': 14, 'sofa': 18,
           'train': 19, 'tvmonitor': 20}

def read_class_set(root, cls, dataset, phase):
    if dataset == 'pascal':
        setname = cls + '_' + phase + '.txt'
        setpath = os.path.join(
            root, 'PASCAL', 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', setname
        )
    elif dataset == 'imagenet':
        setname = '_'.join([cls, dataset, phase]) + '.txt'
        setpath = os.path.join(root, 'Image_sets', setname)
    else:
        raise ValueError('Unknown {} dataset'.format(dataset))

    if dataset == 'pascal':
        def gen_set(lines):
            for line in lines:
                words = line.split()
                if words[1] == '-1':
                    continue

                item = np.array(
                    [(cls, dataset, words[0])],
                    dtype=[('class', 'U16'), ('dataset', 'U16'), ('imgid', 'U16')]
                )
                yield item

    else:
        def gen_set(lines):
            for line in lines:
                item = np.array(
                    [(cls, dataset, line)],
                    dtype=[('class', 'U16'), ('dataset', 'U16'), ('imgid', 'U16')]
                )
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
        # TODO Convert the bounding box from one- to zero-based
        return '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
            CLASSES[self.cls], 1.0, *self.bbox,
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
            6 - math.floor(self.azimuth * 3.5/180)
        )

class Annotations(object):
    def __init__(self, root, cls, dataset, imgid, exclude_occluded=False):
        self.__root__ = os.path.normpath(root)
        matpath = os.path.join(
            self.__root__, 'Annotations', cls + '_' + dataset, imgid + '.mat'
        )
        data = sio.loadmat(matpath)
        record = data['record'][0][0]

        if dataset == 'pascal':
            imgname = record['imgname'][0]
            self.__imgname__ = os.path.join('PASCAL', 'VOCdevkit', imgname)
        elif dataset == 'imagenet':
            filename = record['filename'][0]
            self.__imgname__ = os.path.join(
                'Images', cls + '_' + dataset, filename
            )
        else:
            raise ValueError('Unknown {} dataset'.format(dataset))

        self.__imgsize__ = record['imgsize'][0]
        self.__objects__ = record['objects'][0]
        self.__data__ = self.read_data(cls, exclude_occluded)

    def read_data(self, cls, exclude_occluded):
        data = []

        for obj in list(self.__objects__):
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

            boxes = self.overlapping_boxes(obj['bbox'][0])
            def create_pose(box):
                return Pose(
                    cls=cls, bbox=box, azimuth=azimuth, elevation=elevation,
                    theta=theta
                )
            poses = np.apply_along_axis(create_pose, 1, boxes)
            data.append(poses)

        return data

    def overlapping_boxes(self, bbox):
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
        boxes = np.rint(boxes).astype(np.int)
        return boxes

    def tolines(self):
        imgpath = os.path.join(self.__root__, self.__imgname__)
        lines = '{}\n{}\n{}\n{}\n'.format(imgpath, *self.__imgsize__[[2, 0, 1]])

        if self.__data__:
            poses = np.concatenate(self.__data__)
            lines += '{}\n'.format(poses.size)
            lines += ''.join(map(lambda x: x.toline(), poses))
        else:
            lines += '0\n'

        return lines
