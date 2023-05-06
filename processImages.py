import os
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

## Lists all of the images in the data folder
file_path = "data1Cropped/"
image_list = os.listdir(file_path)

## Initializes an empty matrix for the images to be put in
img_matrix = np.empty([901*1201,len(image_list)])

index = 0
print("Compressing images into one matrix")

for image in image_list:
    ## Opens an image and converts it to greyscale
    grey_img = Image.open(file_path + image).convert("L")
    ## Converts the datatype of the image and saves it as a numpy array
    pixel_array = np.asarray(grey_img).astype(np.uint8)
    
    ## Finds the height and width of the image (1201,901)
    width = len(pixel_array[0])
    height = len(pixel_array)
    
    ## Reshapes the image array to be a column vector that is 901*1201 long
    flattened_array = np.reshape(pixel_array.T,[901*1201])
    
    ## Inserts the column vector as a column of the large image matrix
    img_matrix[:,index] = flattened_array
    index+=1
    
print("Matrix shape: " + str(img_matrix.shape))
print("Number of images in dataset: " + str(len(img_matrix[0,:])))
print("Performing SVD")

## Performs SVD on the large image matrix
## Setting full_matrices=False is kind of like "econ" in Matlab
u, s, v = np.linalg.svd(img_matrix,full_matrices=False)

print("S shape: " + str(s.shape))
print("U shape: " + str(u.shape))
print("V shape: " + str(v.shape))
print("Reconstructing faces")

## Reconstruct faces with different numbers of components
for num_comps in range(1,53):
    ## Remakes the large image matrix by multiplying U*S*V'
    ## np.matrix converts U,S,V to a matrix  that can be multiplied easier
    remade = np.matrix(u[:,:num_comps]) * np.diag(s[:num_comps]) * np.matrix(v[:num_comps, : ])
    
    ## Selects a random person to show
    personToShow = random.randint(0,53)
    
    ## Extracts the random column and reshapes it to original size
    reshaped = remade[:,personToShow].reshape(901,1201)
    
    ## Converts the numpy matrix to an Image and rotates it
    new_image = Image.fromarray(reshaped).rotate(270, Image.NEAREST, expand = 1)
    
    ## Converts the image to be RGB
    ## Image is still greyscale, but this just changed how it is stored
    if new_image.mode != 'RGB':
        new_image = new_image.convert('RGB')
        
    ## Saves image to a new folder with the component name
    new_image.save("diffPeople/" + str(num_comps) + "_comps.jpg")
    print("Saved image: " + "diffPeople/" + str(num_comps) + "_comps.jpg")


## Show eigenfaces
for num_comps in range(1,5):
    ## Selects a random person to show
    personToShow = random.randint(0,53)
    eigen = np.matrix(u[:, :])* np.diag(s[:])
    print(eigen.shape)
    print(eigen[0,:15])
    face = eigen[:,personToShow]
    reshaped = face.reshape(901,1201)
    mean = np.mean(reshaped)
    reshaped += mean
    new_image = Image.fromarray(reshaped).rotate(270, Image.NEAREST, expand = 1)
    if new_image.mode != 'RGB':
        new_image = new_image.convert('RGB')
    new_image.save("eigenfaces/" + str(personToShow) + "_person.jpg")
   
   
    
## Showing a random face with first 20 prinipal component values of V matrix
for people in range(1,2):
    num_comps = 20
    ## Remakes the large image matrix by multiplying U*S*V'
    ## np.matrix converts U,S,V to a matrix  that can be multiplied easier
    remade = np.matrix(u[:,:num_comps]) * np.diag(s[:num_comps]) * np.matrix(v[:num_comps, : ])
    
    ## Selects a random person to show
    personToShow = random.randint(0,53)
    
    ## Extracts the random column and reshapes it to original size
    reshaped = remade[:,personToShow].reshape(901,1201)
    
    ## Converts the numpy matrix to an Image and rotates it
    new_image = Image.fromarray(reshaped).rotate(270, Image.NEAREST, expand = 1)
    
    ## Converts the image to be RGB
    ## Image is still greyscale, but this just changed how it is stored
    if new_image.mode != 'RGB':
        new_image = new_image.convert('RGB')
        
    ## Saves image to a new folder with the component name
    new_image.save("rightSingular/" + str(personToShow) + "_person.jpg")
    print("Saved image: " + "rightSingular/" + str(personToShow) + "_person.jpg")
    
    ## Plotting first 20 coefficients of V matrix
    plt.plot(list(range(1,num_comps+1)), list(np.array(v[:num_comps, personToShow ])))
    plt.xlabel('Principal Component')
    plt.ylabel('Corresponding Coefficient from V matrix')
    plt.show()