from PIL import Image
import numpy as np

grey_img_1 = Image.open("data1Cropped/img008.jpg").convert("L")
pixel_array_1 = np.asarray(grey_img_1).astype(np.uint8)

grey_img_2 = Image.open("data1Cropped/img008.jpg").convert("L")
pixel_array_2 = np.asarray(grey_img_2).astype(np.uint8)

u1, s1, v1 = np.linalg.svd(pixel_array_1,full_matrices=False)
num_comps = 50
remade1 = np.matrix(u1[:,:num_comps]) * np.diag(s1[:num_comps]) * np.matrix(v1[:num_comps, : ])
# new_image1 = Image.fromarray(remade1)
# new_image1.show()
u2, s2, v2 = np.linalg.svd(pixel_array_2,full_matrices=False)
remade2 = np.matrix(u2[:,:num_comps]) * np.diag(s2[:num_comps]) * np.matrix(v2[:num_comps, : ])
# new_image2 = Image.fromarray(remade2)
# new_image2.show()
flattened_array_1 = np.reshape(np.asarray(remade1),[901*1201])
flattened_array_2 = np.reshape(np.asarray(remade2),[901*1201])
print(flattened_array_1.shape)
print(flattened_array_2.shape)

dot_product = np.dot(flattened_array_1, flattened_array_2)
print(dot_product)
cosine_similarity = dot_product / (np.linalg.norm(flattened_array_1)*np.linalg.norm(flattened_array_2))

print(cosine_similarity)
