#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Configuration/parameters for Pix2Pix-GAN training.
Date   :    15.09.2023
Author :    Eric Einspänner
Mail   :    eric.einspaenner@med.ovgu.de
'''

###############################################################
# * Configuration settings/parameters
###############################################################
cts = ('unhealthy', 'healthy')
data_format = 'npy'  # 'dicom' or 'npy'

### Img properties
# img_shape = (256, 256, 64)
# img_shape = (384, 384, 32)
img_shape = (128, 128, 64)


# grid = (128, 128, 32)
# grid = (128, 128, 16)
grid = (64, 64, 64)


window1 = [(8000, 3000)]
window2 = [(8000, 3000)]
# window1 = [(300, 0)]
# window2 = [(300, 0)]
# window1 = [(2000, 0), (1000, 200), (500, 50)]
# window2 = [(2000, 0), (1000, 200), (500, 50)]


batch_size = 3
epochs = 100


optimizer = 'adam'
learning_rates = (0.00018, 0.1)
L_weights = (1, 100)
sample_interval = 128
model_interval = 1


splitvar = 1.0


resizeconv = True
smoothlabel = False
rescale_intensity = False
coordconv = False
randomshift = 0.1
randomflip = 0.3
gennoise = 0.1

dropout = 0.2
resoutput = 0.0
fmloss = False
multigpu = None


def get_cfg_filename(_img_shape, _grid):
    return "cfg_{}_{}.json".format(_img_shape, _grid)

