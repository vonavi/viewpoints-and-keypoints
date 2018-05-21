import os

import zipfile
import luigi

LIB_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(LIB_PATH, '..', 'data')

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
