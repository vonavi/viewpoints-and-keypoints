import errno
import os

import math
import collections
import numpy as np
import scipy.io as sio

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'plant', 'sheep', 'sofa', 'train', 'tvmonitor']

def annotated_classes():
    annotated_list = [0, 1, 3, 4, 5, 6, 8, 10, 13, 17, 18, 19]
    classes = []
    for idx, cls in enumerate(CLASSES):
        if idx in annotated_list:
            classes.append(cls)
    return classes

class Dataset(object):
    def __init__(self, name, root):
        self.name = name
        self.root = os.path.normpath(root)

    def matpath(self, cls, imgid):
        return os.path.join(
            self.root, 'Annotations', cls + '_' + self.name, imgid + '.mat')

    def read_sets_for_classes(self, setpaths, classes):
        imgid_dict = self.classes_by_imgid(classes)

        records = []
        for path in setpaths:
            with open(path, 'r') as f:
                for line in f.read().splitlines():
                    words = line.split()
                    if len(words) > 1 and words[1] != '1':
                        continue

                    imgid = words[0]
                    classes = imgid_dict[imgid]
                    if not classes:
                        continue

                    item = dict()
                    item['imgid'] = imgid
                    item['classes'] = classes
                    records.append(item)

        return records

    def classes_by_imgid(self, classes):
        cls_dict = dict()
        for cls in classes:
            cls_dir = os.path.join(
                self.root, 'Annotations', cls + '_' + self.name)
            filenames = os.listdir(cls_dir)
            cls_dict[cls] = list(map(lambda f: os.path.splitext(f)[0], filenames))

        imgid_dict = collections.defaultdict(list)
        for cls, indexes in cls_dict.items():
            for imgid in indexes:
                imgid_dict[imgid].append(cls)

        return imgid_dict

class Pascal(Dataset):
    def __init__(self, root):
        super().__init__('pascal', root)

    def imgpath(self, cls, filename):
        imgpath = os.path.join(
            self.root, 'PASCAL', 'VOCdevkit', 'VOC2012', 'JPEGImages', filename)
        if not os.path.isfile(imgpath):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), imgpath)
        return imgpath

    def read_class_set(self, cls, phase):
        setname = cls + '_' + phase + '.txt'
        setpath = os.path.join(
            self.root, 'PASCAL', 'VOCdevkit', 'VOC2012',
            'ImageSets', 'Main', setname)
        return self.read_sets_for_classes([setpath], [cls])

    def read_joint_set(self, classes, phase):
        setname = phase + '.txt'
        setpath = os.path.join(
            self.root, 'PASCAL', 'VOCdevkit', 'VOC2012',
            'ImageSets', 'Main', setname)
        return self.read_sets_for_classes([setpath], classes)

class Imagenet(Dataset):
    def __init__(self, root):
        super().__init__('imagenet', root)

    def imgpath(self, cls, filename):
        imgpath = os.path.join(
            self.root, 'Images', cls + '_' + self.name, filename)
        if not os.path.isfile(imgpath):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), imgpath)
        return imgpath

    def read_class_set(self, cls, phase):
        setname = '_'.join([cls, self.name, phase]) + '.txt'
        setpath = os.path.join(self.root, 'Image_sets', setname)
        return self.read_sets_for_classes([setpath], [cls])

    def read_joint_set(self, classes, phase):
        setpaths = []
        for cls in classes:
            setname = '_'.join([cls, self.name, phase]) + '.txt'
            path = os.path.join(self.root, 'Image_sets', setname)
            setpaths.append(path)
        return self.read_sets_for_classes(setpaths, classes)

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

            objects = record['objects'][0]
            self.read_class_data(cls, objects)

    def read_class_data(self, cls, objects):
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
                    cls=obj_class, bbox=box, azimuth=azimuth,
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

        def gen_boxes():
            for x1_shift in -1, 0, 1:
                for y1_shift in -1, 0, 1:
                    for x2_shift in -1, 0, 1:
                        for y2_shift in -1, 0, 1:
                            x1 = max(bbox[0] + x1_shift * dx, 0)
                            y1 = max(bbox[1] + y1_shift * dy, 0)
                            x2 = min(bbox[2] + x2_shift * dx, self.__width - 1)
                            y2 = min(bbox[3] + y2_shift * dy, self.__height - 1)
                            yield [x1, y1, x2, y2]

        boxes = np.stack(gen_boxes())
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
