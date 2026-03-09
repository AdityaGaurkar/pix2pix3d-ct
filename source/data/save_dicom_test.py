import sys

import numpy as np
import pydicom
import os
import glob
import shutil
from pydicom.uid import ExplicitVRLittleEndian

base_folder = "/home/topf/dicom/ai-dsa/storescp/20*"

f = glob.glob("{}/**/nat".format(base_folder))

for idx, series_nat in enumerate(f):
    print(str(idx+1) + "/" + str(len(f)))
    study_folder = os.path.abspath(os.path.join(series_nat, os.pardir))
    year_folder = os.path.abspath(os.path.join(study_folder, os.pardir))
    accession_series_number = os.path.basename(study_folder)

    nat_study_folder = os.path.join(year_folder, "{}-nat".format(accession_series_number))
    nat_series_folder_fill = os.path.join(nat_study_folder, "fill")
    nat_series_folder_sub = os.path.join(nat_study_folder, "sub")

    if not os.path.exists(nat_series_folder_fill):
        os.makedirs(nat_series_folder_fill)

    if not os.path.exists(nat_series_folder_sub):
        os.makedirs(nat_series_folder_sub)

    for d in os.listdir(series_nat):
        dicom_file = os.path.join(series_nat, d)
        new_fill_file = os.path.join(nat_series_folder_fill, d)
        new_sub_file = os.path.join(nat_series_folder_sub, d)

        if not os.path.exists(new_fill_file):
            shutil.copy(dicom_file, new_fill_file)

        if os.path.exists(new_sub_file):
            continue

        ds = pydicom.dcmread(new_fill_file)
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        x = np.zeros(512 * 512)
        x = x.astype('uint16')
        x_bytes = x.tobytes()
        ds.PixelData = x_bytes
        ds['PixelData'].is_undefined_length = False

        ds.SeriesNumber += 99000
        ds.SeriesInstanceUID += '.98'
        ds.SOPInstanceUID += '.98'

        ds.ImageType = 'DERIVED\\SECONDARY\\AXIAL\\3DANGIO\\SUB'
        ds.SeriesDescription = 'Sub Medium EE Auto Zero [SK]'
        ds.save_as(new_sub_file)

sys.exit(0)

for folder in os.listdir(base_folder):
    folder = os.path.join(base_folder, folder)


input_file = "/home/topf/dicom/ai-dsa/test/167366791-10/fill/1.3.12.2.1107.5.4.7.12002.30000017120608411736500053553"

ds = pydicom.dcmread(input_file)
ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

x = np.arange(0, 512*512) / 100.0

# x = (x - float(ds.RescaleIntercept)) / float(ds.RescaleSlope)

x = x.astype('uint16')
x_bytes = x.tobytes()
ds.PixelData = x_bytes
ds['PixelData'].is_undefined_length = False

ds.SeriesNumber += 99000
ds.SeriesInstanceUID += '.99'
ds.SOPInstanceUID += '.99'

ds.ImageType = 'DERIVED\\SECONDARY\\AXIAL\\3DANGIO\\SUB'
ds.SeriesDescription = 'Sub Medium EE Auto Mo [SK]'

newfile = os.path.join("result", os.path.basename(input_file) + ".99")
ds.save_as(newfile)
