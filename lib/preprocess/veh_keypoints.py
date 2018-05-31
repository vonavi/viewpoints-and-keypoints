import os
import numpy as np
import xml.etree.ElementTree as ET

from preprocess.keypoints import HEAT_MAP_DIMS, Keypoints
from utils.bbox import *

NUM_OF_KEYPOINTS = 27 * 4

class Annotations(object):
    def __init__(self, dataset, imgpath, exclude_occluded=True):
        self.__exclude_occluded = exclude_occluded
        self.__imgpath = imgpath
        self.__data = []

        xmlpath = os.path.splitext(imgpath)[0] + '.xml'
        tree = ET.parse(xmlpath)
        root = tree.getroot()

        size = tree.find('size')
        self.__width = int(size.find('width').text)
        self.__height = int(size.find('height').text)
        self.__depth = int(size.find('depth').text)

        kps_flips = dataset.kps_flips
        objects = root.findall('object')
        self.read_data(objects, kps_flips)

    def read_data(self, objects, kps_flips):
        coords = np.empty((NUM_OF_KEYPOINTS, 2))
        coords.fill(np.nan)

        for obj in objects:
            truncated = bool(int(obj.find('truncated').text))
            difficult = bool(int(obj.find('difficult').text))
            if difficult or (self.__exclude_occluded and truncated):
                continue

            kps_name = obj.find('name').text
            kps_idx = int(kps_name.split('_')[0])

            bndbox = obj.find('bndbox')
            kps_bbox = np.empty(4, dtype=np.int)
            kps_bbox[0] = int(bndbox.find('xmin').text)
            kps_bbox[1] = int(bndbox.find('ymin').text)
            kps_bbox[2] = int(bndbox.find('xmax').text)
            kps_bbox[3] = int(bndbox.find('ymax').text)
            # Make bounding box to be 0-indexed
            kps_bbox -= 1
            if not is_bbox_valid(
                    kps_bbox, width=self.__width, height=self.__height):
                continue

            coords[4 * kps_idx    ] = np.array([kps_bbox[0], kps_bbox[1]])
            coords[4 * kps_idx + 1] = np.array([kps_bbox[0], kps_bbox[3]])
            coords[4 * kps_idx + 2] = np.array([kps_bbox[2], kps_bbox[1]])
            coords[4 * kps_idx + 3] = np.array([kps_bbox[2], kps_bbox[3]])

        image_bbox = np.array([0, 0, self.__width - 1, self.__height - 1])
        for box in bbox_overlaps(image_bbox):
            keypoints = Keypoints(
                class_idx=0, bbox=box, coords=coords,
                start_idx=0, kps_flips=kps_flips)
            self.__data.append(keypoints)

    def is_empty(self):
        return len(self.__data) == 0

    def tolines(self):
        lines = '{}\n{}\n{}\n{}\n{}\n'.format(
            self.__imgpath, self.__depth, self.__height, self.__width,
            NUM_OF_KEYPOINTS * HEAT_MAP_DIMS[0] * HEAT_MAP_DIMS[1])
        lines += '{}\n'.format(len(self.__data))
        lines += ''.join(map(lambda x: x.toline(), self.__data))
        return lines
