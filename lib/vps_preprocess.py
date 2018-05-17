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

class CollectClassData(luigi.Task):
    cls = luigi.ChoiceParameter(choices=pascal3d.CLASSES, var_type=str)
    phase = luigi.ChoiceParameter(choices=['train', 'val'], var_type=str)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(CACHE_PATH, 'vps_' + self.cls, self.phase + '.txt'))

    def requires(self):
        return UnzipPascal3d()

    def run(self):
        def write_annotations(dataset, imgset):
            for idx, item in enumerate(list(imgset)):
                cls = item['class'][0]
                imgid = item['imgid'][0]

                f.write('# {}\n'.format(idx))
                annot = pascal3d.Annotations(
                    cls=cls, dataset=dataset, imgid=imgid)
                f.write(annot.tolines())

        pascal3d_root = self.input().path
        with self.output().open('w') as f:
            pascal = pascal3d.Pascal(pascal3d_root)
            pascal_set = pascal.read_class_set(self.cls, self.phase)
            write_annotations(pascal, pascal_set)

            imagenet = pascal3d.Imagenet(pascal3d_root)
            imagenet_set = imagenet.read_class_set(self.cls, self.phase)
            write_annotations(imagenet, imagenet_set)

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
