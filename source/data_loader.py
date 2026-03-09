#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
DataLoader class for Pix2Pix-GAN training.
Version:    1.0
Date   :    15.09.2023
Author :    Eric Einspänner
Mail   :    eric.einspaenner@med.ovgu.de
'''
########################################################################
# * Import
########################################################################
import ast
import itertools
import numbers
import numpy as np
import os
import pickle
import pydicom
from skimage import exposure
import threading
import queue
import math
# import multiprocessing


########################################################################
# * Functions
########################################################################
def WND(X, W):
    r"""
    Apply the windowing operation to an input tensor (image).

    :param X: Input tensor.
    :param W: Window parameters.

    :return: Windowed tensor.
    """
    R = 255. * (X - W[1] + 0.5 * W[0]) / W[0]
    R[R < 0] = 0
    R[R > 255] = 255
    return R


def rWND(R, W):
    r"""
    Reverse the windowing operation on an input tensor.

    :param R: Windowed tensor.
    :param W: Window parameters.

    :return: Reverted tensor.
    """
    X = R * W[0] / 255. + W[1] - 0.5 * W[0]
    return X


########################################################################
# * DataLoader Class
########################################################################
class MyDataLoader():
    def __init__(self, df, cts=('VNC', 'DE'), img_shape=(512, 512, 8), grid=(1, 1, 1),
                 window1=[(2000, 0)], window2=[(2000, 0)], rescale_intensity=False, splitvar=0.8
                 ):
        self.cts = cts
        self.img_shape = img_shape
        self.grid = grid
        self.window1 = window1
        self.window2 = window2
        self.rescale_intensity = rescale_intensity
        self.splitvar = splitvar
        self._volume_cache = {}

        def slice_count(x):
            r"""
            Calculate slice position and number based on image position in a DataFrame.

            :param x: Input DataFrame containing 'ImagePosition(Patient)' column.

            :return: DataFrame with added 'slice_pos' (slice position) and 'slice_num' (slice number) columns.
            """
            if 'slice_num' in x.columns:
                x['slice_num'] = x['slice_num'].astype(int)
                M = x['slice_num'].max()
                m = x['slice_num'].min()
                if M == m:
                    x['slice_pos'] = 0.0
                else:
                    x['slice_pos'] = (x['slice_num'] - m) / (M - m)
                return x

            if 'zpos' in x.columns:
                x['xx'] = x['zpos'].astype(float)
            elif 'ImagePosition(Patient)' in x.columns:
                x['xx'] = x['ImagePosition(Patient)'].apply(
                    lambda v: [str(n).strip() for n in ast.literal_eval(v)][-1]).astype(float)
            else:
                raise ValueError("DataFrame must contain either 'slice_num', 'zpos', or 'ImagePosition(Patient)'")
            
            # Calculate maximum and minimum values of 'xx'
            M = x['xx'].max()
            m = x['xx'].min()
            
            # Calculate 'slice_pos' based on 'xx' values
            if M == m:
                x['slice_pos'] = 0.0
            else:
                x['slice_pos'] = (M - x['xx']) / (M - m)
            
            # Calculate 'slice_num' based on ranking of 'xx' values
            x['slice_num'] = x['xx'].rank(method='min', ascending=False).astype(int) - 1
            
            # Remove the intermediate 'xx' column
            del x['xx']
            
            return x

        # Preprocesses the input DataFrame and initializes class attributes
        self.df = df.reset_index(drop=True)
        self.df = self.df.groupby(['pid', 'ct'])
        self.df = self.df.apply(slice_count)
        self.df = self.df.reset_index(drop=True)
        self.df = self.df.sort_values(by=['pid', 'ct', 'slice_num'])
        self.df = self.df.reset_index(drop=True)
        first_row = self.df.iloc[0]
        tpid = first_row['pid']
        tfilepath = first_row['filepath']
        self.rescale_in = float(first_row.get('RescaleIntercept', 0.0))
        self.rescale_sl = float(first_row.get('RescaleSlope', 1.0))
        first_arr = self._read_slice_file(tfilepath)
        first_hwd = self._to_hwd(first_arr)
        self.rows = int(first_hwd.shape[0])
        self.cols = int(first_hwd.shape[1])
        self.volume_mode = 'Depth' in self.df.columns or first_hwd.shape[2] > 1 and \
            self.df.groupby(['pid', 'ct']).size().max() == 1

        tsplit = tfilepath.split(os.sep)
        self.basedir = os.path.join(*tsplit[:tsplit.index(tpid)])

        qstring = 'ct=="' + self.cts[0] + '"'
        if 'Depth' in self.df.columns:
            dff = self.df.query(qstring).groupby('pid')['Depth'].max().astype(int)
        else:
            dff = self.df.query(qstring).groupby('pid')['slice_num'].max() + 1
        # print(dff)
        self.case_list = [(k, dff[k]) for k in dff.index.tolist()]
        # print(self.case_list)

        # split train/non-train sets
        self.case_split = None
        self.split()

        # get total_samples
        self.total_samples = self.get_total_samples()


    @staticmethod
    def _read_slice_file(filepath):
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.npy':
            arr = np.load(filepath)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = np.squeeze(arr, axis=-1)
            return arr
        return pydicom.dcmread(filepath).pixel_array


    def _to_hwd(self, arr):
        if arr.ndim == 2:
            return np.expand_dims(arr, axis=-1)
        if arr.ndim != 3:
            raise ValueError("Expected 2D or 3D array, got shape {}".format(arr.shape))

        # Already H, W, D
        if arr.shape[0] >= arr.shape[2] and arr.shape[1] >= arr.shape[2]:
            return arr

        # D, H, W -> H, W, D
        return np.moveaxis(arr, 0, -1)


    def _load_volume(self, pid, ct):
        key = (pid, ct)
        if key in self._volume_cache:
            return self._volume_cache[key]

        files = self.df[(self.df['pid'] == pid) & (self.df['ct'] == ct)]['filepath'].tolist()
        if not files:
            raise ValueError("No files found for pid='{}', ct='{}'".format(pid, ct))

        if self.volume_mode and len(files) == 1:
            vol = self._to_hwd(self._read_slice_file(files[0]))
        else:
            slices = [self._read_slice_file(x) for x in files]
            slices = [np.squeeze(s) if np.ndim(s) == 3 and s.shape[-1] == 1 else s for s in slices]
            for s in slices:
                if np.ndim(s) != 2:
                    raise ValueError("Expected 2D slice file in slice mode for pid='{}', ct='{}'".format(pid, ct))
            vol = np.moveaxis(np.array(slices), 0, -1)

        vol = vol.astype(float) * self.rescale_sl + self.rescale_in
        self._volume_cache[key] = vol
        return vol


    def split(self):
        r"""
        Split the list of cases into training and non-training sets.

        This method either performs a random split or loads a predefined split from a file.

        :return: None
        """
        N = len(self.case_list)
        s = np.full(N, False)

        # random split or load from split.pkl
        if isinstance(self.splitvar, numbers.Number):
            # Randomly choose a subset of cases for the training set
            choose = np.random.choice(N, size=int(self.splitvar * N), replace=False)
            s[choose] = True

            # Store the split in case_split attribute
            self.case_split = []
            self.case_split.append(list(itertools.compress(self.case_list, s)))
            self.case_split.append(list(itertools.compress(self.case_list, ~s)))
        else:
            # Load the split from a pickle file
            with open(self.splitvar, 'rb') as f:
                self.case_split = pickle.load(f)


    def save_split(self, savepath):
        r"""
        Save the split information to a pickle file.

        :param savepath: Path to the pickle file where the split information will be saved.

        :return: None
        """
        with open(savepath, 'wb') as f:
            # Serialize and write the case_split list to the pickle file
            pickle.dump(self.case_split, f)


    def get_total_samples(self):
        r"""
        Generate and return a list of arrays containing total sample information for each case in the split.

        :return: List of arrays, where each array represents the sample information for a specific case.
        """
        # Initialize an empty list to store arrays of sample information
        A = []

        # Get the grid parameters for sampling
        gr, gc, gz = self.grid

        # Iterate over each case in the split
        for c in self.case_split:
            # Initialize an empty array for case-specific sample information
            C = np.array([], dtype='uint16').reshape(0, 4)

            # Iterate over the cases within the current split
            for i, case in enumerate(c):
                _, tz = case

                # Generate grids of x, y, and z indices for sampling
                x, y, z = np.meshgrid(
                    range(1 + (self.rows - self.img_shape[0]) // gr),
                    range(1 + (self.cols - self.img_shape[1]) // gc),
                    range(1 + (tz - self.img_shape[2]) // gz)
                )

                # Create an array containing sample information: [case_index, x_index, y_index, z_index]
                B = np.moveaxis(np.array([gr * x, gc * y, gz * z]), 0, -1).reshape(-1, 3).astype('uint16')
                # Add case index to each row
                k = np.full((B.shape[0], 1), i, dtype='uint16')
                # Concatenate case index and sample indices
                B = np.concatenate((k, B), axis=-1)
                # Concatenate the new sample information to the case-specific array
                C = np.concatenate((C, B), axis=0)
            
            # Append the case-specific array to the list
            A.append(C)

        # Return the list of arrays containing total sample information
        return A


    def load_dicoms(self, pid, slice_nums, window=False):
        r"""
        Load paired image volumes (DICOM or NPY slice files) for a specific patient and slice range.

        :param pid: Patient ID or index.
        :param slice_nums: Tuple containing the start and end slice numbers.
        :param window: Flag to apply windowing to loaded images.

        :return: Tuple of two 3D arrays representing loaded DICOM images for two contrast types.
        """
        if isinstance(pid, int):
            # Get the patient ID using the provided index
            pid = self.case_list[pid][0]

        slice_num_start, slice_num_end = slice_nums

        B_full = self._load_volume(pid, self.cts[0])
        A_full = self._load_volume(pid, self.cts[1])
        A = A_full[:, :, slice_num_start:slice_num_end]
        B = B_full[:, :, slice_num_start:slice_num_end]

        if window:
            # Apply windowing using window parameters for contrast types
            A = WND(A, self.window2[0])
            B = WND(B, self.window1[0])

        # Return the loaded DICOM images as a tuple of 3D arrays
        return A, B


    def imread(self, case, pos, window=True, split=0):
        r"""
        Load and preprocess images for a specific case and position.

        :param case: Index of the case.
        :param pos: Tuple containing the position (rx, ry, rz).
        :param window: Flag to apply windowing to loaded images.
        :param split: Index of the split (train or test) to use.

        :return: Tuple of preprocessed 3D arrays representing images for two contrast types.
        """
        rx, ry, rz = pos
        pid, zs = self.case_split[split][case]

        xm = rx
        xM = xm + self.img_shape[1]
        ym = ry
        yM = ym + self.img_shape[0]

        slice_num_start = rz
        slice_num_end = rz + self.img_shape[2]

        # Load DICOM images for the specified patient and slice range
        A, B = self.load_dicoms(pid, (slice_num_start, slice_num_end))
        A = A[ym:yM, xm:xM, :]
        B = B[ym:yM, xm:xM, :]

        if window:
            a = []
            b = []
            for w in self.window2:
                # Apply windowing using window parameters for contrast type 1
                a.append(WND(A, w))
            for w in self.window1:
                # Apply windowing using window parameters for contrast type 0
                b.append(WND(B, w))
            A = np.stack(a, axis=-1)
            B = np.stack(b, axis=-1)

        # Return preprocessed images as a tuple of 3D arrays
        return A, B


    # same as imread (see above)
    def imread_slice(self, case, pos, window=True, split=0):
        r"""
        Load and preprocess images for a specific slice within a case and position.

        :param case: Index of the case.
        :param pos: Tuple containing the position (rx, ry, rz).
        :param window: Flag to apply windowing to loaded images.
        :param split: Index of the split (train or test) to use.

        :return: Tuple of preprocessed 3D arrays representing images for two contrast types.
        """
        rx, ry, rz = pos
        pid, zs = self.case_split[split][case]

        xm = rx
        xM = xm + self.img_shape[1]
        ym = ry
        yM = ym + self.img_shape[0]

        slice_num_start = rz
        slice_num_end = rz + self.img_shape[2]

        A, B = self.load_dicoms(pid, (slice_num_start, slice_num_end))
        A = A[ym:yM, xm:xM, :]
        B = B[ym:yM, xm:xM, :]

        if window:
            a = []
            b = []
            for w in self.window2:
                a.append(WND(A, w))
            for w in self.window1:
                b.append(WND(B, w))
            A = np.stack(a, axis=-1)
            B = np.stack(b, axis=-1)

        return A, B


    def load_batch(self, batch_size=1, split=0):
        r"""
        Load a batch of preprocessed image pairs for training.

        :param batch_size: Batch size.
        :param split: Index of the split (train or test) to use.

        :return: Generator yielding a batch of preprocessed image pairs for each iteration.
        """
        total_samples = self.total_samples[split]
        #print('Total samples: ', total_samples)
        self.n_batches = int(len(total_samples) / batch_size)
        #print('n_batches', self.n_batches)

        # Sample random indices from total_samples for each batch
        # Sample n_batches * batch_size from each path list so that the model sees all
        # samples from both domains
        points = np.random.choice(len(total_samples), size=self.n_batches * batch_size, replace=True) # replace=False

        buffer_size = 16
        q = queue.Queue(buffer_size)

        thread_size = 8

        class ProducerThread(threading.Thread):
            def __init__(self, data_loader: MyDataLoader, index: int, no_threads: int):
                super(ProducerThread, self).__init__()
                self.data_loader = data_loader
                self.index = index
                self.no_threads = no_threads

            def run(self):
                for idx in range(int(math.ceil(self.data_loader.n_batches / self.no_threads))):
                    i = idx * self.no_threads + self.index

                    if i >= self.data_loader.n_batches:
                        break

                    batch_p = points[i * batch_size:(i + 1) * batch_size]
                    imgs_A, imgs_B = [], []
                    for p in batch_p:
                        #print('DataLoader Total samples: ',total_samples[p])
                        c = total_samples[p][0]
                        pos = total_samples[p][1:]

                        img_A, img_B = self.data_loader.imread(c, pos)
                        if self.data_loader.rescale_intensity:
                            img_A = exposure.rescale_intensity(img_A, out_range=(-0.95, 0.95))
                            img_B = exposure.rescale_intensity(img_B, out_range=(-0.95, 0.95))
                        else:
                            img_A = img_A / 127.5 - 1.
                            img_B = img_B / 127.5 - 1.
                        imgs_A.append(img_A)
                        imgs_B.append(img_B)

                    imgs_A = np.array(imgs_A)
                    imgs_B = np.array(imgs_B)
                    q.put((imgs_A, imgs_B), block=True)

        # Create and start producer threads
        for i in range(thread_size):
            p = ProducerThread(data_loader=self, index=i, no_threads=thread_size)
            p.start()

        # Yield batches of preprocessed image pairs for training
        for i in range(self.n_batches):
            d = q.get(block=True)
            yield d


    def load_data(self, batch_size=1, split=0):
        r"""
        Load and preprocess a batch of image pairs for testing.

        :param batch_size: Batch size.
        :param split: Index of the split (train or test) to use.

        :return: Tuple containing preprocessed image arrays for domains A and B.
        """
        total_samples = self.total_samples[split]
        batches = np.random.choice(len(total_samples), size=batch_size, replace=False)

        imgs_A = []
        imgs_B = []
        for b in batches:
            c = total_samples[b][0]
            pos = total_samples[b][1:]
            img_A, img_B = self.imread(c, pos, split=split)
            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A) / 127.5 - 1.
        imgs_B = np.array(imgs_B) / 127.5 - 1.

        return imgs_A, imgs_B
