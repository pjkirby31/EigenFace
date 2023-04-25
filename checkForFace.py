import os
from PIL import Image
import numpy as np
import random

grey_img = Image.open("data1Cropped/img003.jpg").convert("L")
pixel_array = np.asarray(grey_img).astype(np.uint8)
flattened_array = np.reshape(pixel_array.T,[901*1201]).astype(np.uint8)

u, s, v = np.linalg.svd(pixel_array,full_matrices=False)

print("S shape: " + str(s.shape))
print("U shape: " + str(u.shape))
print("V shape: " + str(v.shape))