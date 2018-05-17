import os

from ftplib import FTP
import math
import zipfile
import numpy as np
import luigi

from preprocess import pascal3d

LIB_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(LIB_PATH, '..', 'data')
CACHE_PATH = os.path.join(LIB_PATH, '..', 'cachedir')

def write_annotations(dataset, imgset, fd):
    for idx, item in enumerate(list(imgset)):
        imgid = item['imgid'][0]
        classes = item['classes'][0]

        fd.write('# {}\n'.format(idx))
        img_annot = pascal3d.Annotations(
            classes=classes, dataset=dataset, imgid=imgid)
        fd.write(img_annot.tolines())

class CollectJointData(luigi.Task):
    phase = luigi.ChoiceParameter(choices=['train', 'val'], var_type=str)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(CACHE_PATH, 'vps_joint', self.phase + '.txt'))

    def requires(self):
        return UnzipPascal3d()

    def run(self):
        pascal3d_root = self.input().path
        classes = pascal3d.annotated_classes()
        with self.output().open('w') as fd:
            pascal = pascal3d.Pascal(pascal3d_root)
            pascal_set = pascal.read_joint_set(classes, self.phase)
            write_annotations(pascal, pascal_set, fd)

            imagenet = pascal3d.Imagenet(pascal3d_root)
            imagenet_set = imagenet.read_joint_set(classes, self.phase)
            write_annotations(imagenet, imagenet_set, fd)

class CollectClassData(luigi.Task):
    annotated_classes = pascal3d.annotated_classes()
    cls = luigi.ChoiceParameter(choices=annotated_classes, var_type=str)
    phase = luigi.ChoiceParameter(choices=['train', 'val'], var_type=str)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(CACHE_PATH, 'vps_' + self.cls, self.phase + '.txt'))

    def requires(self):
        return UnzipPascal3d()

    def run(self):
        pascal3d_root = self.input().path
        with self.output().open('w') as fd:
            pascal = pascal3d.Pascal(pascal3d_root)
            pascal_set = pascal.read_class_set(self.cls, self.phase)
            write_annotations(pascal, pascal_set, fd)

            imagenet = pascal3d.Imagenet(pascal3d_root)
            imagenet_set = imagenet.read_class_set(self.cls, self.phase)
            write_annotations(imagenet, imagenet_set, fd)

class UnzipPascal3d(luigi.Task):
    def output(self):
        return luigi.LocalTarget(os.path.join(DATA_PATH, 'PASCAL3D'))

    def requires(self):
        return DownloadPascal3d()

    def run(self):
        zip_path = self.input().path
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(DATA_PATH)
        os.rename(os.path.splitext(zip_path)[0], self.output().path)

class DownloadPascal3d(luigi.Task):
    filename = 'PASCAL3D+_release1.1.zip'

    def output(self):
        return luigi.LocalTarget(
            os.path.join(DATA_PATH, self.filename), format=luigi.format.Nop)

    def run(self):
        ftp = FTP('cs.stanford.edu')
        ftp.login()
        ftp.cwd('cs/cvgl')

        ftp.voidcmd('TYPE I')
        conn, total_size = ftp.ntransfercmd('RETR ' + self.filename)
        out_file = self.output().open('w')
        size_written = 0

        while True:
            block = conn.recv(8192)
            if not block:
                break

            out_file.write(block)
            size_written += len(block)
            self.set_status_message(
                'Progress: {} / {}'.format(size_written, total_size))
            self.set_progress_percentage(
                math.floor(100 * size_written / total_size))

        conn.close()
        out_file.close()

if __name__ == '__main__':
    luigi.run()
