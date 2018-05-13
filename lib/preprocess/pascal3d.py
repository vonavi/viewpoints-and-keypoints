import os

import numpy as np

CLASSES = {'aeroplane': 1, 'bicycle': 2, 'boat': 4, 'bottle': 5, 'bus': 6,
           'car': 7, 'chair': 9, 'diningtable': 11, 'motorbike': 14, 'sofa': 18,
           'train': 19, 'tvmonitor': 20}

def convert_set(root, cls, dataset, imgset):
    if dataset == 'pascal':
        setname = cls + '_' + imgset + '.txt'
        setpath = os.path.join(
            root, 'PASCAL', 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', setname
        )
    elif dataset == 'imagenet':
        setname = '_'.join([cls, dataset, imgset]) + '.txt'
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
