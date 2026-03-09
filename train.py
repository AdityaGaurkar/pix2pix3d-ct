
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
import re
import datetime
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

try:
    import mlflow
except Exception:
    mlflow = None


def normalize_cfg_paths(cfg, train_dir, output_dir):
    """Resolve legacy absolute paths from older configs across OSes."""
    df_train = cfg.get('df_train')
    if isinstance(df_train, str):
        df_train = df_train.replace('\\', os.sep)
        fallback_df = os.path.join(train_dir, 'select.ftr')
        if os.path.exists(df_train):
            cfg['df_train'] = df_train
        elif os.path.exists(fallback_df):
            print("Resolved missing df_train path to:", fallback_df)
            cfg['df_train'] = fallback_df

    splitvar = cfg.get('splitvar')
    if isinstance(splitvar, str):
        splitvar = splitvar.replace('\\', os.sep)
        fallback_split = os.path.join(output_dir, 'split.pkl')
        if os.path.exists(splitvar):
            cfg['splitvar'] = splitvar
        elif os.path.exists(fallback_split):
            print("Resolved missing splitvar path to:", fallback_split)
            cfg['splitvar'] = fallback_split
        else:
            print("Configured split file missing. Falling back to random split:", c.splitvar)
            cfg['splitvar'] = c.splitvar


def _mlflow_param_value(v):
    if isinstance(v, (str, int, float, bool)):
        return v
    if v is None:
        return "None"
    return json.dumps(v)


########################################################################
# * Configuration
########################################################################
### define paths
base_dir = r'/home/cet/pix2pix/pix2pix3d-ct/rat_data'
train_dir = r'/home/cet/pix2pix/pix2pix3d-ct/rat_data/train'
output_dir = base_dir + '/result'


### load config OR create a new one
cfg_path = os.path.join(output_dir, c.get_cfg_filename(c.img_shape, c.grid))

if os.path.exists(cfg_path):
    print("Loading existing config: {}".format(cfg_path))
    with open(cfg_path) as json_file:
        cfg = json.load(json_file)
    normalize_cfg_paths(cfg, train_dir, output_dir)
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

debug_input_dir = os.path.join(output_dir, 'debug_input')
DL.dump_preprocessed_sample(debug_input_dir, split=0, sample_index=0)
print("Saved preprocessed debug tensors to:", debug_input_dir)

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
        m = re.search(r'(\d+)$', str(epoch))
        epoch = int(m.group(1)) if m else 0
    else:
        epoch = 0
else:
    epoch = 0
    print("No trained model found in {}. Start from scratch!".format(models_dir))


########################################################################
# * Train
########################################################################
mlflow_active = mlflow is not None
if mlflow_active:
    tracking_dir = os.path.join(output_dir, "mlruns")
    os.makedirs(tracking_dir, exist_ok=True)
    mlflow.set_tracking_uri("file://" + os.path.abspath(tracking_dir))
    mlflow.set_experiment("pix2pix3d-ct")
    run_name = datetime.datetime.now().strftime("train-%Y%m%d-%H%M%S")
    mlflow.start_run(run_name=run_name)
    mlflow.set_tags({
        "project": "pix2pix3d-ct",
        "stage": "train",
        "data_format": str(cfg.get("data_format", "unknown")),
    })
    mlflow.log_params({k: _mlflow_param_value(v) for k, v in cfg.items()})
    mlflow.log_param("epoch_start", int(epoch))
    mlflow.log_param("train_cases", int(len(DL.case_split[0])))
    mlflow.log_param("val_cases", int(len(DL.case_split[1])))
else:
    print("MLflow not installed; proceeding without experiment tracking.")

def _metric_logger(metrics_dict, step):
    if mlflow_active:
        mlflow.log_metrics(metrics_dict, step=int(step))

run_status = "FINISHED"
try:
    gan.train(epochs=cfg['epochs'], batch_size=cfg['batch_size'], sample_interval=cfg['sample_interval'],
              model_interval=cfg['model_interval'], epoch_start=epoch, metric_logger=_metric_logger)
    ########################################################################
    # * Plot
    ########################################################################
    utils.plot_tracking_gan(output_dir)
except Exception:
    run_status = "FAILED"
    raise
finally:
    if mlflow_active:
        artifacts = [
            os.path.join(output_dir, c.get_cfg_filename(c.img_shape, c.grid)),
            os.path.join(output_dir, "log.txt"),
            os.path.join(output_dir, "loss.png"),
            split_path,
        ]
        for artifact_path in artifacts:
            if os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path)
        debug_dir = os.path.join(output_dir, "debug_input")
        if os.path.isdir(debug_dir):
            mlflow.log_artifacts(debug_dir, artifact_path="debug_input")
        mlflow.end_run(status=run_status)
