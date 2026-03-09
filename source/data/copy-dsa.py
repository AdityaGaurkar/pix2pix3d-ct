import glob
import os
import shutil
import progressbar

import numpy as np
import pydicom
from PIL import Image

folder = "/home/topf/dicom/ai-dsa/storescp/20*/*/fill/"
output_folder = "/home/topf/dicom/ai-dsa/storescp/ausschnitt-both3"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def window_level(arr):
    arr = np.array(arr, dtype=float)
    arr[arr > 2500] = 2500
    arr[arr < 500] = 500
    arr -= 500

    arr = arr / 2000 * 255

    return arr.astype(np.uint8)


folders = [f for f in glob.glob(folder) if f.find("nat") < 1]

pb = progressbar.ProgressBar(max_value=len(folders))

for idx, f in enumerate(folders):
    pb.update(idx+1)
    number = 4

    acc_no = os.path.basename(os.path.realpath(os.path.join(f, os.path.pardir)))
    all_fill_files = np.array(os.listdir(f))
    all_fill_files.sort()

    all_sub_files = np.array(os.listdir(os.path.join(f, os.path.pardir, "sub")))
    all_sub_files.sort()

    if set(all_fill_files) != set(all_sub_files):
        print("Differing files for sub and fill, skip:", acc_no)
        continue

    indexes = np.arange(int(len(all_fill_files) / (number + 1)), len(all_fill_files), int(len(all_fill_files) / (number + 1)))
    indexes = indexes[0:number]
    # print(indexes)

    files = [os.path.join(f, filename) for filename in all_fill_files[indexes]]
    files += [os.path.join(f.replace("fill", "sub"), filename) for filename in all_fill_files[indexes]]

    number *= 2

    arr = np.zeros(shape=(1024, 512*int(number/2)), dtype=np.uint8)

    for idx, file in enumerate(files):
        ds = pydicom.dcmread(file)
        row = 0 if idx < number / 2 else 1
        col = idx % (int(number/2))

        arr[row*512:(row+1)*512, col*512:(col+1)*512] = window_level(ds.pixel_array)
        arr[:, (col+1)*512-1:(col+1)*512] = 255

    arr[512:513, :] = 255

    im = Image.fromarray(arr)
    im.save(os.path.join(output_folder, "{}.png".format(acc_no)))

    # shutil.copy(f, os.path.join(output_folder, os.path.basename(os.path.realpath(os.path.join(f, os.path.pardir, os.path.pardir)))))
