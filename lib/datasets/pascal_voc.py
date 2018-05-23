import errno
import os

import collections

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
        self.segkps_dir = segkps_dir

    def imgpath(self, _, filename):
        imgpath = os.path.join(
            self.root, 'PASCAL', 'VOCdevkit', 'VOC2012', 'JPEGImages', filename)
        if not os.path.isfile(imgpath):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), imgpath)
        return imgpath

    def segkps_path(self, cls):
        return os.path.join(self.segkps_dir, cls + '.mat')

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
