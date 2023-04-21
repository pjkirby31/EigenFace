import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

file_path = "data1Cropped/"
image_list = os.listdir(file_path)
img_matrix = np.empty([901*1201,len(image_list)])
# print(img_matrix.shape)
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
    
# print(img_matrix.shape)
print("Matrix shape: " + str(img_matrix.shape))
print(len(img_matrix[0,:]))
print("Performing SVD")
u, s, v = np.linalg.svd(img_matrix,full_matrices=False)
print("S shape: " + str(s.shape))
print("U shape: " + str(u.shape))
print("V shape: " + str(v.shape))
# weight = (U*S) \ pixelVec
# recon = U*S*weight + meanCol
#  final = reshape(recon, [1201,901])
print("Reconstructing faces")
for num_comps in range(1,53):

    # reshaped = np.matrix(u[:,0]).reshape(901,1201)
    # print(reshaped.shape)
    remade = np.matrix(u[:,:num_comps]) * np.diag(s[:num_comps]) * np.matrix(v[:num_comps, : ])
    # remade = np.matrix(u[:,:]) * np.diag(s[:]) * np.matrix(v[:num_components, : ])
    print(remade.shape) ## np.matrix(u[:, :num_components])
    reshaped = remade[:,0].reshape(901,1201)
    plt.gray()
    plt.imshow(reshaped)
    # os.path.join("/components", str(num_comps),"_comps.png")    
    plt.savefig("components/" + str(num_comps) + "_comps.png")