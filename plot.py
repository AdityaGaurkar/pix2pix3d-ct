#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Plot training tracking OR prediction metrics after training/prediction.
Version:    1.0
Date   :    15.09.2023
Author :    Eric Einspänner
Mail   :    eric.einspaenner@med.ovgu.de
'''
########################################################################
# * Import
########################################################################
import utils


########################################################################
# * Main
########################################################################
if __name__ == '__main__':
    # input_dir = 'C:/Users/einspaen/AppData/Local/xnat-dataset/__Results__/Model_230907'
    # utils.plot_tracking_gan(input_dir)

    input_dirs = ['C:/Users/einspaen/AppData/Local/xnat-dataset/__Results__/Model_230907/TESTDIRECTORY',
                  'C:/Users/einspaen/AppData/Local/xnat-dataset/__Results__/Model_230907/TESTDIRECTORY2']
    utils.plot_2_metrics(input_dirs)