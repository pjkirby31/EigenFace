from PIL import Image
import numpy as np

## Determines whether or not to show images
show_images = False
num_comps = 50

## Opens images and converts them to greyscale
grey_img_1 = Image.open("data1Cropped/img003.jpg").convert("L")
pixel_array_1 = np.asarray(grey_img_1).astype(np.uint8)

grey_img_2 = Image.open("data1Cropped/img002.jpg").convert("L")
pixel_array_2 = np.asarray(grey_img_2).astype(np.uint8)

## Performs SVD on both images separately
## Then reconstructs the images with a set number of components
u1, s1, v1 = np.linalg.svd(pixel_array_1,full_matrices=False)
remade1 = np.matrix(u1[:,:num_comps]) * np.diag(s1[:num_comps]) * np.matrix(v1[:num_comps, : ])

u2, s2, v2 = np.linalg.svd(pixel_array_2,full_matrices=False)
remade2 = np.matrix(u2[:,:num_comps]) * np.diag(s2[:num_comps]) * np.matrix(v2[:num_comps, : ])

## Shows the images if you want to see who they are
if show_images:
    new_image1 = Image.fromarray(remade1)
    new_image2 = Image.fromarray(remade2)
    new_image1.show()
    new_image2.show()
    
## Reshapes the images so they are column vectors
flattened_array_1 = np.reshape(np.asarray(remade1),[901*1201])
flattened_array_2 = np.reshape(np.asarray(remade2),[901*1201])

## Performs cosine similarity by finding angle between vectors
dot_product = np.dot(flattened_array_1, flattened_array_2)
cosine_similarity = dot_product / (np.linalg.norm(flattened_array_1)*np.linalg.norm(flattened_array_2))

print("Cosine similarity after " + str(num_comps) + " component SVD: " + str(round(cosine_similarity,3)*100) + "%")


flattened_array_1 = np.reshape(np.asarray(pixel_array_1),[901*1201])
flattened_array_2 = np.reshape(np.asarray(pixel_array_2),[901*1201])

## Performs cosine similarity by finding angle between vectors
dot_product = np.dot(flattened_array_1, flattened_array_2)
cosine_similarity = dot_product / (np.linalg.norm(flattened_array_1)*np.linalg.norm(flattened_array_2))

print("Cosine similarity of raw images:          "  + str(round(cosine_similarity,3)*100) + "%")
print("Actual value: " + str(cosine_similarity * 100) + "%")