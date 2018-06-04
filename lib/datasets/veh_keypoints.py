import os
import csv
import numpy as np

NUM_OF_KEYPOINTS = 27 * 4

class VehKeypoints(object):
    def __init__(self, root):
        self.__root = os.path.normpath(root)
        self.parts = self.get_parts(self.__root)
        self.kps_flips = self.keypoint_flips(self.parts)

    @staticmethod
    def get_parts(root):
        parts_path = os.path.join(root, 'veh_keypoints.csv')

        with open(parts_path, 'r') as f:
            parts_reader = csv.reader(f, delimiter=',')
            # Skip the vehicle-keypoints header
            next(parts_reader)

            parts_dict = dict()
            for idx_str, kps_name in parts_reader:
                idx = int(idx_str)
                parts_dict[idx] = kps_name

        parts = []
        for _, kps_name in sorted(parts_dict.items()):
            parts.append(kps_name)
        return parts

    @staticmethod
    def keypoint_flips(parts):
        kps_flips = np.empty(NUM_OF_KEYPOINTS, dtype=np.int)

        flip_dict = dict()
        for idx, part_name in enumerate(parts):
            left_to_right = part_name.replace('Left', 'Right')
            right_to_left = part_name.replace('Right', 'Left')

            if left_to_right != part_name:
                flip_dict[left_to_right] = idx
            elif right_to_left != part_name:
                flip_dict[right_to_left] = idx
            else:
                flip_dict[part_name] = idx

        for kps_idx, part_name in enumerate(parts):
            flip_kps_idx = flip_dict[part_name]
            kps_flips[4 * kps_idx    ] = 4 * flip_kps_idx + 2
            kps_flips[4 * kps_idx + 1] = 4 * flip_kps_idx + 3
            kps_flips[4 * kps_idx + 2] = 4 * flip_kps_idx
            kps_flips[4 * kps_idx + 3] = 4 * flip_kps_idx + 1

        return kps_flips

    def get_image_paths(self):
        image_paths = []
        for d, _, files in os.walk(self.__root):
            img_filenames = [img for img in files if img.endswith('.jpg')]
            xml_filenames = [xml for xml in files if xml.endswith('.xml')]

            for filename in img_filenames:
                xml = os.path.splitext(filename)[0] + '.xml'
                if xml in xml_filenames:
                    image_paths.append(os.path.join(d, filename))

        return image_paths

    def read_set(self, image_file):
        with open(image_file, 'r') as f:
            imgset = f.read().splitlines()
        return imgset
