import shutil
import sys
import os
import re

import progressbar
import pydicom
import glob
import numpy as np
from pydicom.uid import ExplicitVRLittleEndian
import multiprocessing


folder = "/home/topf/dicom/ai-dsa/storescp/20*/*/*/*"

files = glob.glob(folder)
print(len(files))

buffer_size = 16
q = multiprocessing.Queue(len(files))
for f in files:
    q.put(f)

pb = progressbar.ProgressBar(max_value=len(files))


class ProducerThread(multiprocessing.Process):
    def __init__(self, queue: multiprocessing.Queue, index: int, no_threads: int, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(ProducerThread, self).__init__()
        self.index = index
        self.no_threads = no_threads
        self.target = target
        self.name = name
        self.queue = queue

    def run(self):
        while not self.queue.empty():
            file = self.queue.get()
            pb.update(len(files) - self.queue.qsize())

            ds = pydicom.dcmread(file, stop_before_pixels=True)

            acc_no = str(ds.AccessionNumber)
            series_number = str(ds.SeriesNumber)

            if not acc_no.endswith("-{}".format(series_number)):
                ds = pydicom.dcmread(file)
                ds.AccessionNumber = "{}-{}".format(acc_no, series_number)
                ds.save_as(file)

            if ds.file_meta.TransferSyntaxUID != ExplicitVRLittleEndian:
                ds = pydicom.dcmread(file)
                x = ds.pixel_array
                # x = x.astype('uint16')
                x_bytes = x.tobytes()

                ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

                ds.PixelData = x_bytes

                ds['PixelData'].is_undefined_length = False
                ds.save_as(file)

            shutil.move(file, os.path.join(os.path.dirname(file), "{:04d}.dcm".format(ds.InstanceNumber)))


thread_size = 16

for i in range(thread_size):
    p = ProducerThread(queue=q, index=i, no_threads=thread_size, name='producer-{}'.format(i))
    p.start()
