import sys
import numpy as np

path = sys.argv[1]
arr = np.load("/home/cet/pix2pix/pix2pix3d-ct/rat_data/train/Rat-Maxillae-1/healthy/6510_right_side_16.npy")

print("path:", path)
print("shape:", arr.shape)
print("dtype:", arr.dtype)
print("min:", float(arr.min()))
print("max:", float(arr.max()))
print("mean:", float(arr.mean()))
print("std:", float(arr.std()))

is_01 = arr.min() >= 0 and arr.max() <= 1
print("looks_normalized_0_1:", is_01)

if is_01:
    model_input = arr * 2.0 - 1.0
else:
    model_input = arr / 127.5 - 1.0

print("model_input_min:", float(model_input.min()))
print("model_input_max:", float(model_input.max()))
print("model_input_mean:", float(model_input.mean()))