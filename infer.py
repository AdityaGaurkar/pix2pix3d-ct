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
import sys
import traceback
import re
import pandas as pd
import utils

from source.data_loader import MyDataLoader
from source.my3dpix2pix import My3dPix2Pix

import tensorflow as tf
try:
    import mlflow
    mlflow_import_error = None
except Exception:
    mlflow = None
    mlflow_import_error = traceback.format_exc()


def load_inference_config(search_roots):
    if isinstance(search_roots, str):
        search_roots = [search_roots]

    json_files = []
    for root in search_roots:
        if not root:
            continue
        json_files.extend(glob.glob(os.path.join(root, '**', 'cfg_*.json'), recursive=True))
        json_files.extend(glob.glob(os.path.join(root, '*.json')))
    json_files = sorted(set(json_files))
    if not json_files:
        raise FileNotFoundError("No JSON config found in {}".format(search_roots))

    # Deterministic selection: use most recently modified config.
    json_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    cfg_path = json_files[0]
    print("Using config:", cfg_path)
    with open(cfg_path) as json_file:
        return json.load(json_file), cfg_path


def resolve_mlflow_context_dir(spath, cfg_path):
    # Prefer config directory (usually the exact training output dir).
    candidates = [
        os.path.dirname(cfg_path),
        spath,
        os.path.dirname(spath),
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return spath


def find_resume_run_id(context_dir):
    # 1) Explicit env override.
    env_id = os.environ.get("MLFLOW_RUN_ID")
    if env_id:
        return env_id

    # 2) Standard marker in context dir.
    run_id_file = os.path.join(context_dir, "mlflow_last_run_id.txt")
    if os.path.exists(run_id_file):
        with open(run_id_file, "r") as f:
            return f.read().strip()

    # 3) One level up fallback (when infer/train paths differ by one level).
    parent_run_id_file = os.path.join(os.path.dirname(context_dir), "mlflow_last_run_id.txt")
    if os.path.exists(parent_run_id_file):
        with open(parent_run_id_file, "r") as f:
            return f.read().strip()

    # 4) shared fallback used by training script.
    shared_run_id_file = "/home/cet/pix2pix/mlflow_last_run_id.txt"
    if os.path.exists(shared_run_id_file):
        with open(shared_run_id_file, "r") as f:
            return f.read().strip()

    return None


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
            "{}/dice".format(prefix): float(m['dice']),
        }, step=i)
    mlflow.log_metrics({
        "{}/psnr_mean".format(prefix): float(np.mean([x['psnr'] for x in metrics_list])),
        "{}/ssim_mean".format(prefix): float(np.mean([x['ssim'] for x in metrics_list])),
        "{}/nmse_mean".format(prefix): float(np.mean([x['nmse'] for x in metrics_list])),
        "{}/dice_mean".format(prefix): float(np.mean([x['dice'] for x in metrics_list])),
        "{}/n_cases".format(prefix): float(len(metrics_list)),
    })


def _experiment_name_from_context_dir(path):
    name = os.path.basename(os.path.normpath(path))
    m = re.match(r"model_(\d+)$", name)
    if m:
        return "model{}".format(m.group(1))
    return name or "pix2pix3d-ct"


def _model_output_prefix(path):
    name = os.path.basename(os.path.normpath(path))
    return name or "model"


########################################################################
# * Test data
########################################################################
### configuration
base_dir = r"/home/cet/pix2pix/pix2pix3d-ct/rat_data"
test_dir = r"/home/cet/pix2pix/pix2pix3d-ct/rat_data/test"

# load config
models_root = os.environ.get("PIX2PIX_MODELS_ROOT", "/home/cet/pix2pix/models")
legacy_root = r"/home/cet/pix2pix/pix2pix3d-ct/rat_data/result"
cfg, cfg_path = load_inference_config([models_root, legacy_root])
model_dir = os.path.dirname(cfg_path)
mlflow_context_dir = resolve_mlflow_context_dir(model_dir, cfg_path)

mlflow_active = mlflow is not None
print("Python executable:", sys.executable)
mlflow_resumed_run = False
if mlflow_active:
    tracking_dir = os.environ.get("MLFLOW_TRACKING_DIR", "/home/cet/pix2pix/mlruns")
    os.makedirs(tracking_dir, exist_ok=True)
    mlflow.set_tracking_uri("file://" + os.path.abspath(tracking_dir))
    resume_run_id = find_resume_run_id(mlflow_context_dir)
    if resume_run_id:
        print("Resuming MLflow run:", resume_run_id)
        # Do not set experiment before resuming by run_id.
        mlflow.start_run(run_id=resume_run_id)
        mlflow_resumed_run = True
        resumed_exp_id = mlflow.active_run().info.experiment_id
        resumed_exp = mlflow.get_experiment(resumed_exp_id)
        mlflow_experiment = resumed_exp.name if resumed_exp is not None else str(resumed_exp_id)
    else:
        mlflow_experiment = "pix2pix3d-ct"
        mlflow.set_experiment(mlflow_experiment)
        run_name = _model_output_prefix(mlflow_context_dir)
        mlflow.start_run(run_name=run_name)
        print("No training run id found; started new infer run.")
    print("MLflow tracking dir:", os.path.abspath(tracking_dir))
    print("MLflow experiment:", mlflow_experiment)
    mlflow.set_tags({
        "project": "pix2pix3d-ct",
        "stage": "infer",
        "has_inference_metrics": "true",
    })
else:
    print("MLflow not installed; proceeding without experiment tracking.")
    if mlflow_import_error:
        print(mlflow_import_error)

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
gan = My3dPix2Pix(DL, savepath=model_dir, L_weights=cfg['L_weights'], opt=cfg['opt'], lrs=cfg['lrs'],
                  smoothlabel=cfg['smoothlabel'], fmloss=cfg['fmloss'],
                  gennoise=cfg['gennoise'],
                  randomshift=cfg['randomshift'], resoutput=cfg['resoutput'], dropout=cfg['dropout'],
                  coordconv=cfg['coordconv'], resizeconv=cfg['resizeconv'], multigpu=cfg['multigpu'])

# tf.keras.utils.plot_model(gan.combined, to_file="my_model.png", show_shapes=True)

# Load final weights
loaded_weights = gan.load_final_weights()
if not loaded_weights:
    raise FileNotFoundError("No trained weights found in {}".format(os.path.join(model_dir, "models")))
print("Loaded weights:", loaded_weights)
if mlflow_active:
    if not mlflow_resumed_run:
        mlflow.log_params({k: _mlflow_param_value(v) for k, v in cfg.items()})
        mlflow.log_param("loaded_weights", str(loaded_weights))
    # Params are immutable on resumed runs; use tags for inference metadata.
    mlflow.set_tag("infer_loaded_weights", str(loaded_weights))
    mlflow.set_tag("infer_data_format", str(cfg.get("data_format", "unknown")))
    mlflow.set_tag("infer_cts", str(cfg.get("cts", "")))

model_output_prefix = _model_output_prefix(mlflow_context_dir)

## make directory for test results inside result/YOURFOLDER
savedir = gan.make_directory('{}_infer'.format(model_output_prefix))
print("Inference output dir:", savedir)
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
        gan = My3dPix2Pix(DL, savepath=model_dir, L_weights=cfg['L_weights'], opt=cfg['opt'], lrs=cfg['lrs'],
                          smoothlabel=cfg['smoothlabel'], fmloss=cfg['fmloss'],
                          gennoise=cfg['gennoise'],
                          randomshift=cfg['randomshift'], resoutput=cfg['resoutput'], dropout=cfg['dropout'],
                          coordconv=cfg['coordconv'], resizeconv=cfg['resizeconv'], multigpu=cfg['multigpu'])

        loaded_weights = gan.load_final_weights()
        if not loaded_weights:
            raise FileNotFoundError("No trained weights found in {}".format(os.path.join(model_dir, "models")))
        print("Loaded weights:", loaded_weights)

        savedir = gan.make_directory('{}_infer_additional'.format(model_output_prefix))
        print("Inference additional output dir:", savedir)
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
