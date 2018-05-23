import os

import math
import zipfile
import luigi

from vps_preprocess import UnzipPascal3d
from datasets.pascal_voc import Dataset, Pascal
from preprocess import segkps

LIB_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(LIB_PATH, '..', 'data')
CACHE_PATH = os.path.join(LIB_PATH, '..', 'cachedir')

def write_annotations(task, dataset, imgset):
    img_total = len(imgset)
    img_idx = 0

    with task.output().open('w') as f:
        for item in imgset:
            img_annot = segkps.Annotations(
                classes=item['classes'], dataset=dataset, imgid=item['imgid'])

            if not img_annot.is_empty():
                f.write('# {}\n'.format(img_idx))
                f.write(img_annot.tolines())
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

class CreateKpsJoint(luigi.Task):
    phase = luigi.ChoiceParameter(choices=['train', 'val'], var_type=str)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(CACHE_PATH, 'kps_joint', self.phase + '.txt'))

    def requires(self):
        return [UnzipPascal3d(), UnzipSegKps()]

    def run(self):
        pascal3d_root = self.input()[0].path
        segkps_dir = self.input()[1].path
        classes = Dataset.annotated_classes()

        pascal = Pascal(pascal3d_root, segkps_dir)
        pascal_set = pascal.read_joint_set(classes, self.phase)
        write_annotations(self, pascal, pascal_set)

class CreateKpsClass(luigi.Task):
    annotated_classes = Dataset.annotated_classes()
    cls = luigi.ChoiceParameter(choices=annotated_classes, var_type=str)
    phase = luigi.ChoiceParameter(choices=['train', 'val'], var_type=str)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(CACHE_PATH, 'kps_' + self.cls, self.phase + '.txt'))

    def requires(self):
        return [UnzipPascal3d(), UnzipSegKps()]

    def run(self):
        pascal3d_root = self.input()[0].path
        segkps_dir = self.input()[1].path

        pascal = Pascal(pascal3d_root, segkps_dir)
        pascal_set = pascal.read_class_set(self.cls, self.phase)
        write_annotations(self, pascal, pascal_set)

class UnzipSegKps(luigi.Task):
    def segkps_path(self):
        zip_path = self.input().path
        return os.path.splitext(zip_path)[0]

    def output(self):
        return luigi.LocalTarget(self.segkps_path())

    def requires(self):
        return CheckSegKpsZip()

    def run(self):
        with zipfile.ZipFile(self.input().path, 'r') as zip_file:
            zip_file.extractall(self.segkps_path())

class CheckSegKpsZip(luigi.ExternalTask):
    filename = 'segkps.zip'

    def output(self):
        return luigi.LocalTarget(
            os.path.join(DATA_PATH, self.filename), format=luigi.format.Nop)

if __name__ == '__main__':
    luigi.run()
