# Pix2pix3D CT

<p>Keras implementation of modified pix2pix with 3D convolutions. Developed for CT data.</p>
<p>This repository contains the source code of the following paper:</p>
<blockquote>
  <p>
    <strong>Generating Synthetic Contrast Enhancement from Non-contrast Chest Computed Tomography Using a Generative Adversarial Network</strong>
    <br>
    <i>Scientific Reports</i> 2021 Oct 14;11(1):20403. <a href="https://doi.org/10.1038/s41598-021-00058-3">doi: 10.1038/s41598-021-00058-3.</a>
  </p>
</blockquote>

## Notes
<p>To try out your own training and inference, each case in the data set should contain a pair of stack of axial CT scans in DICOM format. Please refer to <code>train.ipynb</code> and <code>inference.ipynb</code> for details.</p>
<p>Although this project was developed for CT data, the pix2pix3D network can work for any type of input data, if <code>source/data_loader.py</code> is properly modified.</p>

### NPY input support
You can also use preprocessed `.npy` slices instead of DICOM by setting `data_format = 'npy'` in `config.py`.

Expected folder layout:
`<dataset_root>/<patient_id>/<ct_name>/*.npy`

Example:
- `.../cnn-train/001/fill/0000.npy`
- `.../cnn-train/001/sub/0000.npy`

Notes:
- You can use either:
- many 2D slice files per CT folder (`*.npy` with shape `(H, W)`), or
- one full 3D volume file per CT folder (`*.npy` with shape `(H, W, D)` or `(D, H, W)`).
- In full-volume mode, keep exactly one `.npy` file in each `<patient>/<ct_name>/` folder.
- `ct_name` folders must match `cts` in `config.py` (default: `('fill', 'sub')`).
- During inference with `.npy` input, generated outputs are saved as `.npy` files.

## Description
This model is built using the Keras library. The provided code includes definitions for the generator, discriminator, and the complete training procedure for the pix2pix model. The model takes two types of images as input: `imgs_A` (original images) and `imgs_B` (conditioning images), and generates fake images as output.

Here's a breakdown of the main components in the code:

**Generator** (``build_generator function``): The generator is a U-Net architecture that takes conditioning images (`imgs_B`) and generates corresponding output images (`fake_A`). It consists of encoder and decoder blocks with convolutional and upsampling layers. The My3dResize class defines a custom layer for 3D resizing.

**Discriminator** (``build_discriminator function``): The discriminator takes pairs of real (original) images (`imgs_A`) and conditioning images (`imgs_B`) or pairs of fake (`fake_A`) and conditioning images (`imgs_B`) as input. It outputs a prediction of whether the images are real or fake. The discriminator is also used to calculate feature maps for feature matching loss (`discriminator_feat`).

**Training** (``train function``): This function handles the training of the pix2pix model. It trains both the generator and discriminator iteratively in a adversarial training setting. The generator aims to generate realistic images that can fool the discriminator, while the discriminator aims to correctly classify real and fake images. The loss functions used include binary cross-entropy loss for the discriminator and a combination of binary cross-entropy loss and mean absolute error (MAE) loss for the generator.

**Data Augmentation**: The code includes data augmentation techniques such as random shifting, random flipping, and adding noise to images.

**Saving and Loading Weights**: The code provides functions for saving and loading the model's weights.

**Sample Images** (s``ample_images function``): This function generates and saves sample images during training. It shows a 3x3 grid of original images (`imgs_B`), generated images (`fake_A`), and ground truth images (`imgs_A`).

**Coordinate Channel Layer** (``_CoordinateChannel class``): This layer is used to append coordinate information to the input images before passing them through the network. It's intended to help the model learn spatial relationships better.

**Loss Functions** (``ssim_mae_loss`` and ``ssim_loss``): These functions define custom loss functions for the generator. ssim_mae_loss is a combination of mean absolute error and structural similarity index (SSIM) loss.
