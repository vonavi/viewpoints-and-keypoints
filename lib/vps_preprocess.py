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

class ConvertPascalTrain(luigi.Task):
    cls = luigi.Parameter()

    def output(self):
        setname = 'pascal_{}_train.npy'.format(self.cls)
        return luigi.LocalTarget(os.path.join(CACHE_PATH, setname))

    def requires(self):
        return UnzipPascal3d()

    def run(self):
        imgset = pascal3d.convert_set(
            self.input().path, self.cls, 'pascal', 'train'
        )
        np.save(self.output().path, imgset)

class ConvertPascalVal(luigi.Task):
    cls = luigi.Parameter()

    def output(self):
        setname = 'pascal_{}_val.npy'.format(self.cls)
        return luigi.LocalTarget(os.path.join(CACHE_PATH, setname))

    def requires(self):
        return UnzipPascal3d()

    def run(self):
        imgset = pascal3d.convert_set(
            self.input().path, self.cls, 'pascal', 'val'
        )
        np.save(self.output().path, imgset)

class ConvertImagenetTrain(luigi.Task):
    cls = luigi.Parameter()

    def output(self):
        setname = 'imagenet_{}_train.npy'.format(self.cls)
        return luigi.LocalTarget(os.path.join(CACHE_PATH, setname))

    def requires(self):
        return UnzipPascal3d()

    def run(self):
        imgset = pascal3d.convert_set(
            self.input().path, self.cls, 'imagenet', 'train'
        )
        np.save(self.output().path, imgset)

class ConvertImagenetVal(luigi.Task):
    cls = luigi.Parameter()

    def output(self):
        setname = 'imagenet_{}_val.npy'.format(self.cls)
        return luigi.LocalTarget(os.path.join(CACHE_PATH, setname))

    def requires(self):
        return UnzipPascal3d()

    def run(self):
        imgset = pascal3d.convert_set(
            self.input().path, self.cls, 'imagenet', 'val'
        )
        np.save(self.output().path, imgset)

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
            os.path.join(DATA_PATH, DownloadPascal3d.filename),
            format=luigi.format.Nop
        )

    def run(self):
        ftp = FTP('cs.stanford.edu')
        ftp.login()
        ftp.cwd('cs/cvgl')

        ftp.voidcmd('TYPE I')
        conn, total_size = ftp.ntransfercmd('RETR ' + DownloadPascal3d.filename)
        out_file = self.output().open('w')
        size_written = 0

        while True:
            block = conn.recv(8192)
            if not block:
                break

            out_file.write(block)
            size_written += len(block)
            self.set_status_message(
                'Progress: {} / {}'.format(size_written, total_size)
            )
            self.set_progress_percentage(
                math.floor(100 * size_written / total_size)
            )

        conn.close()
        out_file.close()

if __name__ == '__main__':
    luigi.run()
