import os
import xml.etree.ElementTree as ET
import numpy as np

from annotations.keypoints import HeatMap
from utils.bbox import is_bbox_valid

NUM_OF_KEYPOINTS = 27 * 4

class Annotations(object):
    def __init__(self, imgpath, exclude_occluded=True):
        self.__exclude_occluded = exclude_occluded
        self.__imgpath = imgpath

        xmlpath = os.path.splitext(imgpath)[0] + '.xml'
        tree = ET.parse(xmlpath)
        root = tree.getroot()

        size = root.find('size')
        self.width = int(size.find('width').text)
        self.height = int(size.find('height').text)
        self.depth = int(size.find('depth').text)

        objects = root.findall('object')
        self.coords = self.__read_coords(objects)

    def __read_coords(self, objects):
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
            if not is_bbox_valid(kps_bbox, width=self.width, height=self.height):
                continue

            coords[4 * kps_idx    ] = np.array([kps_bbox[0], kps_bbox[1]])
            coords[4 * kps_idx + 1] = np.array([kps_bbox[0], kps_bbox[3]])
            coords[4 * kps_idx + 2] = np.array([kps_bbox[2], kps_bbox[1]])
            coords[4 * kps_idx + 3] = np.array([kps_bbox[2], kps_bbox[3]])

        return coords

    def tolines(self):
        return '{}\n{}\n{}\n{}\n{}\n'.format(
            self.__imgpath, self.depth, self.height, self.width,
            NUM_OF_KEYPOINTS * HeatMap.dims[0] * HeatMap.dims[1])
