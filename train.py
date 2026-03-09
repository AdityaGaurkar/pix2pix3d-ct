
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Pix2Pix-GAN training.
Version:    1.0
Date   :    15.09.2023
Author :    Eric Einspänner
Mail   :    eric.einspaenner@med.ovgu.de
'''
########################################################################
# * Import
########################################################################
import json
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from source.data_loader import MyDataLoader
from source.my3dpix2pix import My3dPix2Pix
import utils
import config as c

import tensorflow as tf
tf.test.gpu_device_name()


########################################################################
# * Configuration
########################################################################
### define paths
base_dir = r'D:\AdityaG\pix2pix\pix2pix3d-ct\rat_data'
train_dir = r'D:\AdityaG\pix2pix\pix2pix3d-ct\rat_data\train'
output_dir = base_dir + '/result'


### load config OR create a new one
cfg_path = os.path.join(output_dir, c.get_cfg_filename(c.img_shape, c.grid))

if os.path.exists(cfg_path):
    print("Loading existing config: {}".format(cfg_path))
    with open(cfg_path) as json_file:
        cfg = json.load(json_file)
else:
    print("Creating new config:", cfg_path)
    ## new config
    cfg = {
        'df_train': os.path.join(train_dir, 'select.ftr'),
        'cts': c.cts,
        'data_format': c.data_format,
        'img_shape': c.img_shape,
        'window1': c.window1,
        'window2': c.window2,
        'batch_size': c.batch_size,
        'epochs': c.epochs,
        'opt': c.optimizer,
        'lrs': c.learning_rates,
        'L_weights': c.L_weights,
        'sample_interval': c.sample_interval,
        'model_interval': c.model_interval,
        'grid': c.grid,
        'splitvar': c.splitvar,
        'resizeconv': c.resizeconv,
        'smoothlabel': c.smoothlabel,
        'rescale_intensity': c.rescale_intensity,
        'coordconv': c.coordconv,
        'randomshift': c.randomshift,
        'randomflip' : c.randomflip,
        'gennoise': c.gennoise,
        'dropout': c.dropout,
        'resoutput': c.resoutput,
        'fmloss': c.fmloss,
        'multigpu': c.multigpu,
    }

# print(json.dumps(cfg, indent=2))


### create/load train df
if os.path.exists(cfg["df_train"]):
    print("Reading feather:", cfg['df_train'])
    df_train = pd.read_feather(cfg['df_train'])
else:
    data_format = cfg.get('data_format', 'dicom').lower()
    if data_format == 'npy':
        df_train = utils.my_npys_to_dataframe(train_dir, cfg["cts"])
    else:
        df_train = utils.my_dicoms_to_dataframe(train_dir, cfg["cts"])


### sort and save df
df_train_modify = utils.sort_and_save_dataframe(df_train, train_dir)


########################################################################
# * DataLoader
########################################################################
#df0 = pd.read_feather(cfg['df_train'])
DL = MyDataLoader(df_train_modify, cts=cfg['cts'], img_shape=cfg['img_shape'],
                  grid=cfg['grid'],
                  window1=cfg['window1'], window2=cfg['window2'], rescale_intensity=cfg['rescale_intensity'],
                  splitvar=cfg['splitvar'])


if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

split_path = os.path.join(output_dir, 'split.pkl')
DL.save_split(split_path)
cfg['splitvar'] = split_path

with open(os.path.join(output_dir, c.get_cfg_filename(c.img_shape, c.grid)), 'w') as json_file:
    json.dump(cfg, json_file)


########################################################################
# * GAN
########################################################################
gan = My3dPix2Pix(DL, savepath=output_dir, L_weights=cfg['L_weights'], opt=cfg['opt'], lrs=cfg['lrs'],
                  smoothlabel=cfg['smoothlabel'], fmloss=cfg['fmloss'],
                  gennoise=cfg['gennoise'],
                  randomshift=cfg['randomshift'], resoutput=cfg['resoutput'], dropout=cfg['dropout'],
                  coordconv=cfg['coordconv'], resizeconv=cfg['resizeconv'], multigpu=cfg['multigpu'])

models_dir = os.path.join(output_dir, 'models')
if not os.path.isdir(models_dir):
    os.mkdir(models_dir)

if os.path.exists(models_dir):
    epoch = gan.load_final_weights()
    if epoch:
        epoch = int(epoch)
    else:
        epoch = 0
else:
    epoch = 0
    print("No trained model found in {}. Start from scratch!".format(models_dir))


########################################################################
# * Train
########################################################################
gan.train(epochs=cfg['epochs'], batch_size=cfg['batch_size'], sample_interval=cfg['sample_interval'], model_interval=cfg['model_interval'], epoch_start=epoch)


########################################################################
# * Plot
########################################################################
utils.plot_tracking_gan(output_dir)
