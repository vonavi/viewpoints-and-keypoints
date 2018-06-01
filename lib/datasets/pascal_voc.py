import errno
import os

import collections
import numpy as np
import scipy.io as sio

class Dataset(object):
    CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    ANNOTATED = [0, 1, 3, 4, 5, 6, 8, 10, 13, 17, 18, 19]

    @classmethod
    def annotated_classes(cls):
        classes = []
        for idx, class_name in enumerate(cls.CLASSES):
            if idx in cls.ANNOTATED:
                classes.append(class_name)
        return classes

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
    def __init__(self, root, segkps_dir=None):
        super().__init__('pascal', root)
        if segkps_dir is not None:
            self.__segkps_dir = segkps_dir
            self.records_by_imgid = self.__records_by_imgid()
            parts = self.__get_parts()
            self.start_indexes, self.total_kps = self.__get_start_indexes(parts)
            self.kps_flips = self.__get_keypoints_flips(parts)

    def imgpath(self, _, filename):
        imgpath = os.path.join(
            self.root, 'PASCAL', 'VOCdevkit', 'VOC2012', 'JPEGImages', filename)
        if not os.path.isfile(imgpath):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), imgpath)
        return imgpath

    def segkps_path(self, cls):
        return os.path.join(self.__segkps_dir, cls + '.mat')

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

    def __records_by_imgid(self):
        records_dict = collections.defaultdict(
            lambda: collections.defaultdict(list))
        for cls in self.annotated_classes():
            segkps = sio.loadmat(self.segkps_path(cls))
            keypoints = segkps['keypoints'][0][0]
            for idx, row in enumerate(keypoints['voc_image_id']):
                imgid = row[0][0]
                records_dict[imgid][cls].append(idx)

        return records_dict

    def __get_parts(self):
        parts = []
        for cls in self.CLASSES:
            segkps = sio.loadmat(self.segkps_path(cls))
            keypoints = segkps['keypoints'][0][0]

            def gen_labels():
                for row in keypoints['labels']:
                    for label in row[0]:
                        yield label
            parts.append(np.stack(gen_labels()))

        return parts

    @classmethod
    def __get_start_indexes(cls, parts):
        indexes = np.zeros(len(parts), dtype=np.int)
        start_idx = 0
        for idx in cls.ANNOTATED:
            indexes[idx] = start_idx
            start_idx += len(parts[idx])
        return (indexes, start_idx)

    @staticmethod
    def __get_keypoints_flips(parts):
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
