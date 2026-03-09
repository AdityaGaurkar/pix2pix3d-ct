#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Pix2Pix-GAN prediction.
Version:    1.0
Date   :    15.09.2023
Author :    Eric Einspänner
Mail   :    eric.einspaenner@med.ovgu.de
'''
########################################################################
# * Import
########################################################################
import json
import glob
import numpy as np
import os
import pandas as pd
import utils

from source.data_loader import MyDataLoader
from source.my3dpix2pix import My3dPix2Pix

from pydicom.uid import ExplicitVRLittleEndian

import tensorflow as tf


########################################################################
# * Test data
########################################################################
### configuration
base_dir = r"D:\AdityaG\pix2pix\pix2pix3d-ct\rat_data"
test_dir = r"D:\AdityaG\pix2pix\pix2pix3d-ct\rat_data\test"

# load config
spath = r"D:\AdityaG\pix2pix\models"

json_files = glob.glob(os.path.join(spath, '*.json'))

if json_files:
    with open(json_files[0]) as json_file:
        cfg = json.load(json_file)
else:
    print("No JSON-File found!")

# your own test set and names of ct folders
cfg['df_test'] = os.path.join(test_dir, 'select.ftr')
cfg['cts'] = ('fill', 'sub')
cfg['data_format'] = cfg.get('data_format', 'dicom')
cfg['splitvar'] = 1.0  # fixed

# create/load test df
if os.path.exists(cfg["df_test"]):
    print("Reading feather:", cfg['df_test'])
    df_test = pd.read_feather(cfg['df_test'])
else:
    if cfg.get('data_format', 'dicom').lower() == 'npy':
        df_test = utils.my_npys_to_dataframe(test_dir, cfg["cts"])
    else:
        df_test = utils.my_dicoms_to_dataframe(test_dir, cfg["cts"])

# sort and save df
df_test_modify = utils.sort_and_save_dataframe(df_test, test_dir)

df_test_modify = pd.read_feather(cfg['df_test'])


### DataLoader
DL = MyDataLoader(df_test_modify, cts=cfg['cts'], img_shape=cfg['img_shape'],
                  grid=cfg['grid'],
                  window1=cfg['window1'], window2=cfg['window2'], rescale_intensity=cfg['rescale_intensity'],
                  splitvar=cfg['splitvar'])


### GAN
gan = My3dPix2Pix(DL, savepath=spath, L_weights=cfg['L_weights'], opt=cfg['opt'], lrs=cfg['lrs'],
                  smoothlabel=cfg['smoothlabel'], fmloss=cfg['fmloss'],
                  gennoise=cfg['gennoise'],
                  randomshift=cfg['randomshift'], resoutput=cfg['resoutput'], dropout=cfg['dropout'],
                  coordconv=cfg['coordconv'], resizeconv=cfg['resizeconv'], multigpu=cfg['multigpu'])

# tf.keras.utils.plot_model(gan.combined, to_file="my_model.png", show_shapes=True)

# Load final weights
gan.load_final_weights()

## make directory for test results inside result/YOURFOLDER
savedir = gan.make_directory('TESTDIRECTORY')
split = 0
L = gan.data_loader.case_split[split]
choice = np.arange(len(L))


### generate
for case in choice:
    utils.loop_over_case(gan, L[case], savedir, notruth=False)


### plot
utils.plot_metrics(savedir)


########################################################################
# * Additional test data
########################################################################
### configuration
additional_test_dir = os.path.join(base_dir, 'additional-test')

cfg['df_additional_test'] = os.path.join(additional_test_dir, 'select.ftr')

# create/load additional test df
if os.path.exists(cfg["df_additional_test"]):
    print("Reading feather:", cfg['df_additional_test'])
    df_test = pd.read_feather(cfg['df_additional_test'])
else:
    if cfg.get('data_format', 'dicom').lower() == 'npy':
        df_test = utils.my_npys_to_dataframe(additional_test_dir, cfg["cts"])
    else:
        df_test = utils.my_dicoms_to_dataframe(additional_test_dir, cfg["cts"])

# sort and save df
df_test_modify = utils.sort_and_save_dataframe(df_test, additional_test_dir)

df_test_modify = pd.read_feather(cfg['df_additional_test'])

### DataLoader
DL = MyDataLoader(df_test_modify, cts=cfg['cts'], img_shape=cfg['img_shape'],
                  grid=cfg['grid'],
                  window1=cfg['window1'], window2=cfg['window2'], rescale_intensity=cfg['rescale_intensity'],
                  splitvar=cfg['splitvar'])

### GAN
gan = My3dPix2Pix(DL, savepath=spath, L_weights=cfg['L_weights'], opt=cfg['opt'], lrs=cfg['lrs'],
                  smoothlabel=cfg['smoothlabel'], fmloss=cfg['fmloss'],
                  gennoise=cfg['gennoise'],
                  randomshift=cfg['randomshift'], resoutput=cfg['resoutput'], dropout=cfg['dropout'],
                  coordconv=cfg['coordconv'], resizeconv=cfg['resizeconv'], multigpu=cfg['multigpu'])

# tf.keras.utils.plot_model(gan.combined, to_file="my_model.png", show_shapes=True)

# Load final weights
gan.load_final_weights()

## make directory for test results inside result/YOURFOLDER
savedir = gan.make_directory('TESTDIRECTORY2')
split = 0
L = gan.data_loader.case_split[split]
choice = np.arange(len(L))


### generate
for case in choice:
    utils.loop_over_case(gan, L[case], savedir, notruth=False)


### plot
utils.plot_metrics(savedir)
