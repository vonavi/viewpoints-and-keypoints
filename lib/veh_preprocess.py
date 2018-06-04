import os

import math
import random
import luigi
import numpy as np

from datasets.veh_keypoints import VehKeypoints
from preprocess.keypoints import HeatMap, Keypoints
from annotations.veh_keypoints import Annotations
from utils.bbox import bbox_overlaps

LIB_PATH = os.path.dirname(os.path.realpath(__file__))
CACHE_PATH = os.path.join(LIB_PATH, '..', 'cachedir')
VEH_KEYPOINTS_PATH = os.path.join(LIB_PATH, '..', 'data', 'veh_keypoints')

def dims_to_str(dims):
    if dims[0] == dims[1]:
        return str(dims[0])
    else:
        return '{}x{}'.format(*dims)

def write_annotations(task, dataset, imgset):
    img_total = len(imgset)
    img_idx = 0

    with task.output().open('w') as f:
        for imgpath in imgset:
            img_annot = Annotations(dataset=dataset, imgpath=imgpath)
            img_bbox = np.array([0, 0, img_annot.width - 1, img_annot.height - 1])

            keypoints = []
            for box in bbox_overlaps(img_bbox):
                kps = Keypoints(
                    class_idx=0, bbox=box, coords=img_annot.coords,
                    start_idx=0, kps_flips=dataset.kps_flips)
                keypoints.append(kps)

            if keypoints:
                f.write('# {}\n'.format(img_idx))
                f.write(img_annot.tolines())
                f.write('{}\n'.format(len(keypoints)))
                f.write(''.join(map(lambda x: x.toline(), keypoints)))
                img_idx += 1

                if img_idx % 100 == 0:
                    task.set_status_message(
                        'Progress: {} / {}'.format(img_idx, img_total))
                    task.set_progress_percentage(
                        math.floor(100 * img_idx / img_total))

            else:
                img_total -= 1

        task.set_status_message('Progress: {} / {}'.format(img_total, img_total))
        task.set_progress_percentage(100)

class CreateVehKeypoints(luigi.Task):
    phase = luigi.ChoiceParameter(choices=['train', 'val'], var_type=str)
    heatmap_dims = luigi.TupleParameter(significant=True)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                CACHE_PATH, 'veh_keypoints',
                self.phase + dims_to_str(self.heatmap_dims) + '.txt'))

    def requires(self):
        return TrainValImageSets()

    def run(self):
        if self.phase == 'train':
            imgset_path = self.input()[0].path
        else:
            imgset_path = self.input()[1].path

        HeatMap.dims = self.heatmap_dims
        dataset = VehKeypoints(root=VEH_KEYPOINTS_PATH)
        imgset = dataset.read_set(imgset_path)
        write_annotations(self, dataset, imgset)

class TrainValImageSets(luigi.Task):
    def output(self):
        base_path = os.path.join(CACHE_PATH, 'veh_keypoints_')
        return [luigi.LocalTarget(base_path + 'train.txt'),
                luigi.LocalTarget(base_path + 'val.txt')]

    def requires(self):
        return CheckVehKeypoints()

    def run(self):
        dataset = VehKeypoints(root=VEH_KEYPOINTS_PATH)
        image_paths = dataset.get_image_paths()
        random.shuffle(image_paths)

        train_len = math.ceil(float(3 * len(image_paths)) / float(4))
        train_out, val_out = self.output()
        with train_out.open('w') as f:
            for path in image_paths[:train_len]:
                f.write(path + '\n')
        with val_out.open('w') as f:
            for path in image_paths[train_len:]:
                f.write(path + '\n')

class CheckVehKeypoints(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget(VEH_KEYPOINTS_PATH)

if __name__ == '__main__':
    luigi.run()
