import os
from PIL import Image
import numpy as np
import random

file_path = "data1Cropped/"
image_list = os.listdir(file_path)
img_matrix = np.empty([901*1201,len(image_list)])

index = 0
print("Compressing images into one matrix")

for image in image_list:
    grey_img = Image.open(file_path + image).convert("L")
    pixel_array = np.asarray(grey_img).astype(np.uint8)
    
    width = len(pixel_array[0])
    height = len(pixel_array)
    
    flattened_array = np.reshape(pixel_array.T,[901*1201]).astype(np.uint8)
    img_matrix[:,index] = flattened_array
    index+=1
    
print("Matrix shape: " + str(img_matrix.shape))
print("Number of images in dataset: " + str(len(img_matrix[0,:])))
print("Performing SVD")

u, s, v = np.linalg.svd(img_matrix,full_matrices=False)

print("S shape: " + str(s.shape))
print("U shape: " + str(u.shape))
print("V shape: " + str(v.shape))
print("Reconstructing faces")

for num_comps in range(1,53):
    
    remade = np.matrix(u[:,:num_comps]) * np.diag(s[:num_comps]) * np.matrix(v[:num_comps, : ])
    personToShow = random.randint(0,53)
    reshaped = remade[:,personToShow].reshape(901,1201)
    new_image = Image.fromarray(reshaped).rotate(270, Image.NEAREST, expand = 1)
    if new_image.mode != 'RGB':
        new_image = new_image.convert('RGB')
    new_image.save("diffPeople/" + str(num_comps) + "_comps.jpg")
    print("Saved image: " + "diffPeople/" + str(num_comps) + "_comps.jpg")
