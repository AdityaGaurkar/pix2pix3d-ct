#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Utility functions for preperation of data, training tracking etc. of Pix2Pix-GAN.
Date   :    15.09.2023
Author :    Eric Einspänner
Mail   :    eric.einspaenner@med.ovgu.de
'''
###############################################################
# * Import
###############################################################
import ast
import os
import glob
import math
import pydicom
from pydicom.uid import ExplicitVRLittleEndian
import pandas as pd
import csv
from io import StringIO
import numpy as np
from skimage import metrics
from source.data_loader import WND, rWND

import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

###############################################################
# * Functions
###############################################################
def my_dicoms_to_dataframe(basedir, cts):
    r"""
    Converts DICOM files in specified directories into a pandas DataFrame.
    
    This function reads DICOM files located in the provided base directory and its subdirectories
    based on the specified CT series names (cts). It extracts relevant DICOM header information,
    excluding Pixel Data and certain types of headers, and creates a pandas DataFrame containing
    the extracted data.
    
    Parameters:
        basedir (str): The base directory containing the DICOM files and subdirectories.
        cts (list): A list of CT series names to consider for extracting DICOM files.
        
    Returns:
        pandas.DataFrame: A DataFrame containing extracted DICOM header information.
    """
    print('Start with DICOMs to dataframe...')
    caselist = [os.path.join(basedir, x) for x in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, x))]
    file_list = []
    
    # Collect DICOM file paths from the specified CT series directories
    for x in cts:
        file_list.extend(glob.glob(os.path.join(basedir, '*/' + x + '/*.*')))
    
    # Read a sample DICOM file to extract header information
    tdcmpath = os.path.join(caselist[0], cts[0])
    tdcmpath = [os.path.join(tdcmpath, x) for x in os.listdir(tdcmpath)][0]
    tdcm = pydicom.dcmread(tdcmpath, stop_before_pixels=True)
    
    # Create a list of header names to be extracted
    headers = ['filepath']
    for x in tdcm:
        if x.name == 'Pixel Data' or 'Overlay' in x.name or 'Referring' in x.name or 'Acquisition' in x.name:
            continue
        else:
            name = x.name.replace(' ', '')
            headers.append(name)
    
    # Create a CSV output stream for writing extracted header data
    output = StringIO()
    csv_writer = csv.DictWriter(output, fieldnames=headers)
    csv_writer.writeheader()
    
    # Iterate through DICOM files and extract header data
    for f in file_list:
        #print(f)
        file = pydicom.dcmread(f, stop_before_pixels=True)
        row = {'filepath': f}
        
        for x in file:
            if x.name == 'Pixel Data' or 'Overlay' in x.name or 'Referring' in x.name or 'Acquisition' in x.name:
                continue
            else:
                name = x.name.replace(' ', '')
                row[name] = x.value
        
        unwanted = set(row) - set(headers)
        for unwanted_key in unwanted:
            del row[unwanted_key]
        
        csv_writer.writerow(row)
    
    # Create a pandas DataFrame from the CSV output
    output.seek(0)
    df = pd.read_csv(output, low_memory=False)
    
    # Extract patient ID, CT series name, and z-position from file paths
    df['pid'] = df['filepath'].apply(lambda x: x.split(os.sep)[-3])
    df['ct'] = df['filepath'].apply(lambda x: x.split(os.sep)[-2])
    df['zpos'] = df['ImagePosition(Patient)'].apply(lambda x: [str(n).strip() for n in ast.literal_eval(x)][-1])
    
    # Reorder DataFrame columns
    cols = df.columns.tolist()
    cols = cols[-3:] + cols[:-3]
    df = df[cols]
    
    # Save the DataFrame as a Feather file
    df.to_feather(os.path.join(basedir, 'headers.ftr'))

    print('Finished and saved dataframe to {}.'.format(os.path.join(basedir, 'headers.ftr')))
    
    return df


def my_npys_to_dataframe(basedir, cts):
    r"""
    Builds a dataframe for NPY slice datasets with folder layout:
    basedir/<pid>/<ct>/*.npy
    """
    print('Start with NPYs to dataframe...')

    def _sort_key(path):
        name = os.path.splitext(os.path.basename(path))[0]
        digits = ''.join(ch for ch in name if ch.isdigit())
        if digits:
            return (0, int(digits), name)
        return (1, name)

    rows = []
    for pid in os.listdir(basedir):
        pid_dir = os.path.join(basedir, pid)
        if not os.path.isdir(pid_dir):
            continue
        for ct in cts:
            ct_dir = os.path.join(pid_dir, ct)
            if not os.path.isdir(ct_dir):
                continue
            files = sorted(glob.glob(os.path.join(ct_dir, '*.npy')), key=_sort_key)
            if not files:
                continue

            sample = np.load(files[0], mmap_mode='r')
            if sample.ndim == 3 and sample.shape[-1] != 1:
                # Full-volume mode: one file per ct folder
                if len(files) != 1:
                    raise ValueError(
                        "Found {} npy files in '{}'. For full-volume mode, keep exactly one volume file per ct folder.".format(
                            len(files), ct_dir
                        )
                    )
                arr = np.load(files[0], mmap_mode='r')
                if arr.ndim != 3:
                    raise ValueError("Expected 3D volume npy file: {}".format(files[0]))
                rows.append({
                    'pid': pid,
                    'ct': ct,
                    'zpos': 0.0,
                    'slice_num': 0,
                    'Depth': int(arr.shape[2]),
                    'filepath': files[0],
                    'Rows': int(arr.shape[0]),
                    'Columns': int(arr.shape[1]),
                    'RescaleIntercept': 0.0,
                    'RescaleSlope': 1.0,
                })
                continue

            # Slice mode: one 2D slice per file
            for idx, f in enumerate(files):
                arr = np.load(f, mmap_mode='r')
                if arr.ndim == 3 and arr.shape[-1] == 1:
                    arr = np.squeeze(arr, axis=-1)
                if arr.ndim != 2:
                    raise ValueError("Expected 2D npy slice file: {}".format(f))
                rows.append({
                    'pid': pid,
                    'ct': ct,
                    'zpos': float(idx),
                    'slice_num': int(idx),
                    'filepath': f,
                    'Rows': int(arr.shape[0]),
                    'Columns': int(arr.shape[1]),
                    'RescaleIntercept': 0.0,
                    'RescaleSlope': 1.0,
                })

    if not rows:
        raise ValueError("No .npy files found under '{}' for CT folders {}".format(basedir, cts))

    df = pd.DataFrame(rows)
    print('Finished NPY dataframe.')
    return df


def sort_and_save_dataframe(df, output_directory):
    r"""
    Modifies the given DataFrame by converting 'zpos' column to numeric, sorting the DataFrame
    based on 'pid', 'ct', and 'zpos', then saves the sorted DataFrame to a feather file.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be modified and sorted.
        output_directory (str): The directory path where the sorted DataFrame will be saved.

    Returns:
        pandas.DataFrame: The sorted DataFrame.
    """
    # Convert 'zpos' column to numeric
    df['zpos'] = pd.to_numeric(df['zpos'])
    
    # Sort the DataFrame by 'pid', 'ct', and 'zpos'
    sorted_df = df.sort_values(by=['pid', 'ct', 'zpos'])
    
    # Reset index to create a new DataFrame
    sorted_df_reset = sorted_df.reset_index(drop=True)
    
    # Path to save the sorted DataFrame as a feather file
    output_path = os.path.join(output_directory, 'select.ftr')
    
    # Save the sorted DataFrame to a feather file
    sorted_df_reset.to_feather(output_path)
    
    return sorted_df_reset


def loop_over_case(gan, case, savedir, notruth=False):
    r"""
    Loop over a specific medical imaging case for image generation using a Generative Adversarial Network (GAN).
    
    :Args:
    :param gan: The Generative Adversarial Network used for image generation.
    :param case: A tuple containing patient ID (pid) and the z-stack size (zs) for the medical case.
    :param notruth: If True, generates images without using the ground truth data. Default is False.
    
    This function generates synthetic medical images using a GAN for a specific case specified by 'pid' (patient ID) 
    and 'zs' (z-stack size). The function prepares and processes the input images for the GAN and performs image 
    generation using windows defined in the GAN's data loader. It then saves the generated images and calculates 
    various image quality metrics such as PSNR, SSIM, and NMSE.
    """
    pid, zs = case
    print("Handling", pid)

    dcm_A, dcm_B = gan.data_loader.load_dicoms(pid, (0, zs + 1))
    if notruth:
        dcm_A = np.zeros(dcm_B.shape, dtype=dcm_B.dtype)

    img_size = list(dcm_A.shape)
    gan_size = [gan.img_rows, gan.img_cols, gan.depth]

    print("Image size: {}; GAN-size: {}".format(img_size, gan_size))

    # check/compare the image size to GAN size and skip datasets with smaller images sizes
    if img_size[0] < gan_size[0] or img_size[1] < gan_size[1] or img_size[2] < gan_size[2]:
        print("Skipping {} due to image size smaller than GAN size.".format(pid))
        return  # Skip the rest of the function

    a = []
    b = []
    for w in gan.data_loader.window2:
        a.append(WND(dcm_A, w))
    for w in gan.data_loader.window1:
        b.append(WND(dcm_B, w))
    tot_A = np.stack(a, axis=-1)
    tot_B = np.stack(b, axis=-1)
    tot_A = tot_A.astype('float32') / 127.5 - 1.
    tot_B = tot_B.astype('float32') / 127.5 - 1.

    fakes_raw = np.full((img_size[0], img_size[1], img_size[2]), 0, dtype=tot_B.dtype)
    # counts_raw = np.full((img_size[0], img_size[1], img_size[2]), 0, dtype=int)

    # for zi in range()
    for xi in range(int(math.ceil(img_size[0] / gan_size[0]))):
        x = min(xi * gan_size[0], img_size[0]-gan_size[0])

        for yi in range(int(math.ceil(img_size[1] / gan_size[1]))):
            y = min(yi * gan_size[1], img_size[1]-gan_size[1])

            # for zi in range(img_size[2] + 1 - gan_size[2]):
            for zi in range(int(math.ceil(img_size[2] / gan_size[2]))):
                z = min(zi * gan_size[2], img_size[2] - gan_size[2])
                imgs_B = np.expand_dims(tot_B[x:x+gan_size[0], y:y+gan_size[1], z:z+gan_size[2], :], axis=0)
                fake_A = gan.generator.predict(imgs_B)
                fake_A = 0.5 * fake_A + 0.5
                # fake_A = 255. * fake_A[:, :, :, :, 0]
                fake_A = rWND(255. * fake_A[:, :, :, :, 0], gan.data_loader.window2[0])
                # print(xi, yi, zi, np.min(fake_A), np.max(fake_A), np.mean(fake_A), np.median(fake_A), np.std(fake_A))
                fakes_raw[x:x+gan_size[0], y:y+gan_size[1], z:z+gan_size[2]] = fake_A[0]
                # counts_raw[x:x+gan_size[0], y:y+gan_size[1], zi:zi+gan_size[2]] += 1

    # mcounts = counts_raw.copy()
    # mcounts[mcounts == 0] = 1
    # fakes = np.divide(fakes_raw, mcounts)
    fakes = fakes_raw
    # print(np.min(fakes), np.max(fakes), np.mean(fakes), np.median(fakes), np.std(fakes))

    # # random sample
    # sample = np.random.choice(fakes.shape[-1])
    # sample = np.stack((
    #     dcm_B[:, :, sample].astype(fakes.dtype),
    #     fakes[:, :, sample],
    #     dcm_A[:, :, sample].astype(fakes.dtype)
    # ), axis=-1)

    df1 = gan.data_loader.df
    dcms1 = df1[(df1['pid'] == pid) & (df1['ct'] == gan.data_loader.cts[0])]['filepath'].tolist()

    newpath = os.path.join(savedir, pid)
    if not os.path.isdir(newpath):
        os.mkdir(newpath)
    newpath = os.path.join(newpath, 'generated')
    if not os.path.isdir(newpath):
        os.mkdir(newpath)

    print("Generated image: min={}, max={}".format(np.min(fakes), np.max(fakes)))

    # fakes_int = fakes.astype('int16')
    # print("Generated image: min={}, max={}".format(np.min(fakes_int), np.max(fakes_int)))

    use_npy_output = len(dcms1) > 0 and os.path.splitext(dcms1[0])[1].lower() == '.npy'

    for N, y in enumerate(dcms1):
        x = fakes[:, :, N]

        if use_npy_output:
            newfile = os.path.join(newpath, os.path.splitext(os.path.basename(y))[0] + '.npy')
            np.save(newfile, x.astype(np.float32))
            continue

        ds = pydicom.dcmread(y)
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        x = (x - float(ds.RescaleIntercept)) / float(ds.RescaleSlope)

        x = x.astype('uint16')
        x_bytes = x.tobytes()
        ds.PixelData = x_bytes
        ds['PixelData'].is_undefined_length = False

        ds.SeriesNumber += 99000
        ds.SeriesInstanceUID += '.99'
        ds.SOPInstanceUID += '.99'

        ds.ImageType = 'DERIVED\\SECONDARY\\AXIAL\\3DANGIO\\SUB'
        ds.SeriesDescription = 'Sub Medium EE Auto Mo [SK]'

        newfile = os.path.join(newpath, os.path.basename(y))
        ds.save_as(newfile)

        if False:
            with open(newfile + '.png', 'wb') as f:
                writer = png.Writer(width=x.shape[0], height=x.shape[1], bitdepth=16, greyscale=True)
                zgray2list = x.tolist()
                print("Generated image slice {}: min={}, max={}".format(N, np.min(zgray2list), np.max(zgray2list)))
                writer.write(f, zgray2list)

    if True:
        w_min = (gan.data_loader.window2[0][1] - gan.data_loader.window2[0][0] / 2)
        w_max = (gan.data_loader.window2[0][1] + gan.data_loader.window2[0][0] / 2)
        
        img_original = dcm_A.astype(dtype=float)
        
        print("Window: min={}; max={}".format(w_min, w_max))
        
        img_generated = np.copy(fakes).astype(dtype=float)
        
        img_generated[img_generated > w_max] = w_max
        img_generated[img_generated < w_min] = w_min
        
        #img_generated[img_generated > -100000] = 0
        
        img_original[img_original > w_max] = w_max
        img_original[img_original < w_min] = w_min
        
        print(img_original.shape, img_generated.shape)
        
        print(np.min(img_original), np.max(img_original), np.mean(img_original), np.std(img_original))
        print(np.min(img_generated), np.max(img_generated), np.mean(img_generated), np.std(img_generated))
        
        psnr = metrics.peak_signal_noise_ratio(img_original, img_generated, data_range=int(math.ceil(w_max-w_min)))
        #print("PSNR: {}".format(psnr))
        
        mssim, S = metrics.structural_similarity(img_original, img_generated, data_range=int(math.ceil(w_max-w_min)), full=True, gaussian_weights=True)
        #print("SSIM: {}".format(mssim))
        #print(S.shape)
        
        nmse = np.mean((img_original - img_generated) ** 2, dtype=np.float64) / np.mean((img_original) ** 2, dtype=np.float64)
        #print("NMSE: {}".format(nmse))

        newlog = "[PID: %s] [PSNR: %f] [SSIM: %f] [NMSE: %f]" % (pid, psnr, mssim, nmse)

        print(newlog)
        with open(os.path.join(savedir, 'log.txt'), 'a') as f:
            f.write(newlog + '\n')
    
    return


def plot_tracking_gan(input_dir):
    r"""
    Creates visualizations for tracking loss values during GAN training.
    
    :param input_dir: The path to the directory containing the log file and where the generated plots will be saved.
    """
    ### Extract data from log file and store in lists
    epoch = []
    batch = []
    dloss = []
    dfake = []
    dacc = []
    dfacc = []
    gloss = []
    time = []
    
    with open(os.path.join(input_dir, 'log.txt'), 'r') as f:
        for line in f.readlines():
            line = line.strip()
            e,b,d,a,g,t = line.split('] ')
            e = e.replace('[Epoch ','').split('/')[0]
            b = b.replace('[Batch ','').split('/')[0]
            d, df = d.split(',')[0].replace('[D loss (real): ',''), d.split(',')[1].replace('D loss (fake): ','')
            a, af = a.split(',')[0].replace('[D acc (real): ',''), a.split(',')[1].replace('D acc (fake): ','')
            g = g.replace('[G loss: ','')
            t = t.replace('time: ','')
            
            epoch.append(int(e))
            batch.append(int(b))
            dloss.append(float(d))
            dfake.append(float(df))
            dacc.append(float(a))
            dfacc.append(float(af))
            gloss.append(float(g))
            time.append(t)

    ### Determine the last positions for each epoch
    last_positions = {}
    for idx, num in enumerate(epoch):
        last_positions[num] = idx


    ### Calculate average loss and acc per epoch
    d_loss_avg = []
    g_loss_avg = []
    d_real_acc_avg = []
    d_fake_acc_avg = []

    prev_position = 0  # Start position of the first interval
    for epoch, position in last_positions.items():
        interval_dloss = dloss[prev_position:position]
        interval_dacc = dacc[prev_position:position]
        interval_dfacc = dfacc[prev_position:position]
        interval_gloss = gloss[prev_position:position]

        avg_dloss = sum(interval_dloss) / len(interval_dloss)
        d_loss_avg.append(avg_dloss)

        avg_dacc = sum(interval_dacc) / len(interval_dacc)
        d_real_acc_avg.append(avg_dacc)

        avg_dfacc = sum(interval_dfacc) / len(interval_dfacc)
        d_fake_acc_avg.append(avg_dfacc)

        avg_gloss = sum(interval_gloss) / len(interval_gloss)
        g_loss_avg.append(avg_gloss)

        prev_position = position
    
    ### Calculate average loss and acc per thousand batches
    d_loss_batch = []
    g_loss_batch = []
    d_real_acc_batch = []
    d_fake_acc_batch = []

    group_size = 1000
    for i in range(0, len(dloss), group_size):
        # loss
        dgroup = dloss[i:i+group_size]
        ggroup = gloss[i:i+group_size]
        dmean = sum(dgroup) / len(dgroup)
        gmean = sum(ggroup) / len(ggroup)
        d_loss_batch.append(dmean)
        g_loss_batch.append(gmean)

        # acc
        realgroup = dacc[i:i+group_size]
        fakegroup = dfacc[i:i+group_size]
        realmean = sum(realgroup) / len(realgroup)
        fakemean = sum(fakegroup) / len(fakegroup)
        d_real_acc_batch.append(realmean)
        d_fake_acc_batch.append(fakemean)

    ### Create plots
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 8))

    x1 = np.arange(len(dloss)) + 1
    x2 = np.arange(len(d_loss_avg)) + 1
    x3 = np.arange(len(d_loss_batch)) + 1

    # Subplot 1: dloss and gloss
    ax1.plot(x1, dloss, '-b', linewidth=0.1, label='d_loss')
    ax1.plot(x1, dfake, '-g', linewidth=0.1, label='d_fake')
    ax1.plot(x1, gloss, '-r', linewidth=0.1, label='g_loss')
    ax1.set_ylim(-0.1, 20)
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_xlabel('Iteration')

    # Subplot 2: d_loss_avg and g_loss_avg
    ax2.plot(x2, d_loss_avg, '-b', linewidth=1.1, label='d_loss avg')
    ax2.plot(x2, g_loss_avg, '-r', linewidth=1.1, label='g_loss avg')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    # Subplot 3: d_loss_batch and g_loss_batch
    ax3.plot(x3, d_loss_batch, '-b', linewidth=1.1, label='d_loss batch')
    ax3.plot(x3, g_loss_batch, '-r', linewidth=1.1, label='g_loss batch')
    ax3.set_ylabel('Loss')
    ax3.set_xlabel('thousands of batchs')
    ax3.legend()

    # Subplot 4: d_real_acc_avg and d_fake_acc_avg
    ax4.plot(x2, d_real_acc_avg, '-b', linewidth=1.1, label='d acc (real)')
    ax4.plot(x2, d_fake_acc_avg, '-r', linewidth=1.1, label='d acc (fake)')
    ax4.set_ylabel('Acc')
    ax4.set_xlabel('Epoch')
    ax4.legend()

    # Subplot 5: d_real_acc_batch and d_fake_acc_batch
    ax5.plot(x3, d_real_acc_batch, '-b', linewidth=1.1, label='d acc (real)')
    ax5.plot(x3, d_fake_acc_batch, '-r', linewidth=1.1, label='d acc (fake)')
    ax5.set_ylabel('Acc')
    ax5.set_xlabel('thousands of batchs')
    ax5.legend()

    # save plot
    fig.suptitle('Model Loss')
    plt.tight_layout()
    fig.savefig(os.path.join(input_dir, 'loss.png'))


def plot_metrics(input_dir):
    r"""
    Plots boxplots of model metrics and labels outliers with corresponding patient IDs.
    
    :param input_dir: The directory containing the log file with model metrics.
    
    This function reads model metrics (PSNR, SSIM, NMSE) from a log file in the specified input directory.
    It then creates three boxplots for each metric (PSNR, SSIM, NMSE) and labels the outliers on the plots
    with their corresponding patient IDs. The patient IDs are retrieved from the log file and matched with the
    outlier values. The resulting plots are saved as an image file.
    """
    # View results (log file) and save in lists
    pid = []
    psnr = []
    ssim = []
    nmse = []

    # Read and parse metrics from the log file
    with open(os.path.join(input_dir,'log.txt'), 'r') as f:
        for line in f.readlines():
            line = line.strip()
            id,ps,ss,nm = line.split('] ')
            id = id.replace('[PID: ','')
            ps = ps.replace('[PSNR: ','')
            ss = ss.replace('[SSIM: ','')
            nm = nm.replace('[NMSE: ','').replace(']','')
            
            pid.append(str(id))
            psnr.append(float(ps))
            ssim.append(float(ss))
            nmse.append(float(nm))
    
    # Create boxplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # PSNR boxplot
    psnr_box = axes[0].boxplot([psnr])
    axes[0].set_title('PSNR')
    axes[0].set_xticklabels(['PSNR'])
    
    # SSIM boxplot
    ssim_box = axes[1].boxplot([ssim])
    axes[1].set_title('SSIM')
    axes[1].set_xticklabels(['SSIM'])
    
    # NMSE boxplot
    nmse_box = axes[2].boxplot([nmse])
    axes[2].set_title('NMSE')
    axes[2].set_xticklabels(['NMSE'])
    
    # Add labels for outliers
    def label_outliers(ax, box, data, labels):
        for i, flier in enumerate(box['fliers']):
            x_pos = flier.get_xdata()
            y_pos = flier.get_ydata()
            for x,y in zip(x_pos, y_pos):
                index_of_outlier = data.index(y)
                pid_label = labels[index_of_outlier]
                ax.annotate(pid_label, (x, y), xytext=(0, 10), textcoords="offset points", ha='center')
    
    # Label outliers for each metric
    label_outliers(axes[0], psnr_box, psnr, pid)
    label_outliers(axes[1], ssim_box, ssim, pid)
    label_outliers(axes[2], nmse_box, nmse, pid)
    
    # Save boxplots
    fig.suptitle('Model metrics')
    plt.tight_layout()
    fig.savefig(os.path.join(input_dir, 'metrics.png'))


def plot_2_metrics(input_dirs):
    r"""
    Plots boxplots of model metrics and labels outliers with corresponding patient IDs.
    
    :param input_dir: The directory containing the log file with model metrics.
    
    This function reads model metrics (PSNR, SSIM, NMSE) from a log file in the specified input directory.
    It then creates three boxplots for each metric (PSNR, SSIM, NMSE) and labels the outliers on the plots
    with their corresponding patient IDs. The patient IDs are retrieved from the log file and matched with the
    outlier values. The resulting plots are saved as an image file.
    """
    # Initialize a dictionary to store results
    results = {}

    for input_dir in input_dirs:
        # Initialize lists to store results for this directory
        pid = []
        psnr = []
        ssim = []
        nmse = []

        # Read and parse metrics from the log file
        with open(os.path.join(input_dir,'log.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip()
                id,ps,ss,nm = line.split('] ')
                id = id.replace('[PID: ','')
                ps = ps.replace('[PSNR: ','')
                ss = ss.replace('[SSIM: ','')
                nm = nm.replace('[NMSE: ','').replace(']','')

                pid.append(str(id))
                psnr.append(float(ps))
                ssim.append(float(ss))
                nmse.append(float(nm))

        # Store results in the dictionary
        results[input_dir] = {
            'pid': pid,
            'psnr': psnr,
            'ssim': ssim,
            'nmse': nmse
        }

    # Create boxplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    # PSNR boxplot
    psnr_box = axes[0].boxplot([results[dir]['psnr'] for dir in input_dirs], positions=[1, 1.5])
    axes[0].set_title('PSNR')
    axes[0].set_xlim(0.85, 1.65)
    #axes[0].set_xticks([1, 1.4])
    axes[0].set_xticklabels([r'$Data_{woA}$', r'$Data_{wA}$'])

    # perform t-test
    t_stat, p_value = ttest_ind(results[0]['psnr'], results[1]['psnr'])
    print(f'PSNR - t-statistic: {t_stat}, p-value: {p_value}')

    axes[0].text(1.5, 0.9, f'p-value: {p_value:.4f}', fontsize=8, ha='center')

    # SSIM boxplot
    ssim_box = axes[1].boxplot([results[dir]['ssim'] for dir in input_dirs], positions=[1, 1.5])
    axes[1].set_title('SSIM')
    axes[1].set_xlim(0.85, 1.65)
    #axes[1].set_xticks([1, 1.4])
    axes[1].set_xticklabels([r'$Data_{woA}$', r'$Data_{wA}$'])

    # NMSE boxplot
    nmse_box = axes[2].boxplot([results[dir]['nmse'] for dir in input_dirs], positions=[1, 1.5])
    axes[2].set_title('NMSE')
    axes[2].set_xlim(0.85, 1.65)
    #axes[2].set_xticks([1, 1.4])
    axes[2].set_xticklabels([r'$Data_{woA}$', r'$Data_{wA}$'])

    # Save boxplots
    plt.tight_layout()
    fig.savefig(os.path.join(input_dir, 'compare_metrics.png'))
