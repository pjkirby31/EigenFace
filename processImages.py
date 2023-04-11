import os
from PIL import Image
import numpy as np

file_path = "data1Cropped/"
image_list = os.listdir(file_path)
img_matrix = np.empty([len(image_list),901*1201])
print(img_matrix.shape)
index = 0

for image in image_list:
    grey_img = Image.open(file_path + image).convert("L")
    pixel_array = np.asarray(grey_img).astype(np.uint8)
    
    width = len(pixel_array[0])
    height = len(pixel_array)
    
    flattened_array = np.reshape(pixel_array.T,[1,901*1201]).astype(np.uint8)
    img_matrix[index] = flattened_array
    index+=1
    
print(img_matrix.shape)
u, s, v = np.linalg.svd(img_matrix,full_matrices=False)
print(s)
print(u.shape)