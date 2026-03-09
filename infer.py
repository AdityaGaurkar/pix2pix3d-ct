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
import datetime
import pandas as pd
import utils

from source.data_loader import MyDataLoader
from source.my3dpix2pix import My3dPix2Pix

import tensorflow as tf
try:
    import mlflow
except Exception:
    mlflow = None


def load_inference_config(spath):
    json_files = sorted(glob.glob(os.path.join(spath, '*.json')))
    if not json_files:
        raise FileNotFoundError("No JSON config found in {}".format(spath))

    # Deterministic selection: use most recently modified config.
    json_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    cfg_path = json_files[0]
    print("Using config:", cfg_path)
    with open(cfg_path) as json_file:
        return json.load(json_file)


def dataframe_matches_disk(df, cts):
    if 'filepath' not in df.columns or 'ct' not in df.columns:
        return False
    if not set(cts).issubset(set(df['ct'].unique())):
        return False
    paths = df['filepath'].astype(str).tolist()
    return all(os.path.exists(p) for p in paths)


def _mlflow_param_value(v):
    if isinstance(v, (str, int, float, bool)):
        return v
    if v is None:
        return "None"
    return json.dumps(v)


def _log_case_metrics(metrics_list, prefix):
    if mlflow is None or not metrics_list:
        return
    for i, m in enumerate(metrics_list):
        mlflow.log_metrics({
            "{}/psnr".format(prefix): float(m['psnr']),
            "{}/ssim".format(prefix): float(m['ssim']),
            "{}/nmse".format(prefix): float(m['nmse']),
        }, step=i)
    mlflow.log_metrics({
        "{}/psnr_mean".format(prefix): float(np.mean([x['psnr'] for x in metrics_list])),
        "{}/ssim_mean".format(prefix): float(np.mean([x['ssim'] for x in metrics_list])),
        "{}/nmse_mean".format(prefix): float(np.mean([x['nmse'] for x in metrics_list])),
        "{}/n_cases".format(prefix): float(len(metrics_list)),
    })


########################################################################
# * Test data
########################################################################
### configuration
base_dir = r"/home/cet/pix2pix/pix2pix3d-ct/rat_data"
test_dir = r"/home/cet/pix2pix/pix2pix3d-ct/rat_data/test"

# load config
spath = r"/home/cet/pix2pix/pix2pix3d-ct/rat_data/result"
cfg = load_inference_config(spath)

mlflow_active = mlflow is not None
if mlflow_active:
    tracking_dir = os.path.join(spath, "mlruns")
    os.makedirs(tracking_dir, exist_ok=True)
    mlflow.set_tracking_uri("file://" + os.path.abspath(tracking_dir))
    mlflow.set_experiment("pix2pix3d-ct")
    run_name = datetime.datetime.now().strftime("infer-%Y%m%d-%H%M%S")
    mlflow.start_run(run_name=run_name)
    mlflow.set_tags({
        "project": "pix2pix3d-ct",
        "stage": "infer",
    })
else:
    print("MLflow not installed; proceeding without experiment tracking.")

# your own test set and names of ct folders
cfg['df_test'] = os.path.join(test_dir, 'select.ftr')
cfg['cts'] = ('unhealthy', 'healthy')
cfg['data_format'] = 'npy'
cfg['splitvar'] = 1.0  # fixed

# create/load test df
if os.path.exists(cfg["df_test"]):
    print("Reading feather:", cfg['df_test'])
    df_test = pd.read_feather(cfg['df_test'])
    if not dataframe_matches_disk(df_test, cfg["cts"]):
        print("Cached feather is stale/incompatible. Rebuilding from source files.")
        if cfg.get('data_format', 'dicom').lower() == 'npy':
            df_test = utils.my_npys_to_dataframe(test_dir, cfg["cts"])
        else:
            df_test = utils.my_dicoms_to_dataframe(test_dir, cfg["cts"])
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
loaded_weights = gan.load_final_weights()
if not loaded_weights:
    raise FileNotFoundError("No trained weights found in {}".format(os.path.join(spath, "models")))
print("Loaded weights:", loaded_weights)
if mlflow_active:
    mlflow.log_params({k: _mlflow_param_value(v) for k, v in cfg.items()})
    mlflow.log_param("loaded_weights", str(loaded_weights))

## make directory for test results inside result/YOURFOLDER
savedir = gan.make_directory('TESTDIRECTORY')
split = 0
L = gan.data_loader.case_split[split]
choice = np.arange(len(L))


run_status = "FINISHED"
try:
    ### generate
    main_metrics = []
    for case in choice:
        m = utils.loop_over_case(gan, L[case], savedir, notruth=False)
        if m is not None:
            main_metrics.append(m)

    if mlflow_active:
        _log_case_metrics(main_metrics, "test")


    ### plot
    utils.plot_metrics(savedir)
    if mlflow_active:
        for p in [os.path.join(savedir, "log.txt"), os.path.join(savedir, "metrics.png")]:
            if os.path.exists(p):
                mlflow.log_artifact(p, artifact_path="test")

    ########################################################################
    # * Additional test data
    ########################################################################
    additional_test_dir = os.path.join(base_dir, 'additional-test')

    if os.path.isdir(additional_test_dir):
        cfg['df_additional_test'] = os.path.join(additional_test_dir, 'select.ftr')

        # create/load additional test df
        if os.path.exists(cfg["df_additional_test"]):
            print("Reading feather:", cfg['df_additional_test'])
            df_test = pd.read_feather(cfg['df_additional_test'])
            if not dataframe_matches_disk(df_test, cfg["cts"]):
                print("Cached additional-test feather is stale/incompatible. Rebuilding from source files.")
                if cfg.get('data_format', 'dicom').lower() == 'npy':
                    df_test = utils.my_npys_to_dataframe(additional_test_dir, cfg["cts"])
                else:
                    df_test = utils.my_dicoms_to_dataframe(additional_test_dir, cfg["cts"])
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

        loaded_weights = gan.load_final_weights()
        if not loaded_weights:
            raise FileNotFoundError("No trained weights found in {}".format(os.path.join(spath, "models")))
        print("Loaded weights:", loaded_weights)

        savedir = gan.make_directory('TESTDIRECTORY2')
        split = 0
        L = gan.data_loader.case_split[split]
        choice = np.arange(len(L))

        additional_metrics = []
        for case in choice:
            m = utils.loop_over_case(gan, L[case], savedir, notruth=False)
            if m is not None:
                additional_metrics.append(m)
        if mlflow_active:
            _log_case_metrics(additional_metrics, "additional_test")

        utils.plot_metrics(savedir)
        if mlflow_active:
            for p in [os.path.join(savedir, "log.txt"), os.path.join(savedir, "metrics.png")]:
                if os.path.exists(p):
                    mlflow.log_artifact(p, artifact_path="additional_test")
    else:
        print("Skipping additional test inference; directory not found:", additional_test_dir)
except Exception:
    run_status = "FAILED"
    raise
finally:
    if mlflow_active:
        mlflow.end_run(status=run_status)
