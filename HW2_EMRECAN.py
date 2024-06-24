import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def range_by_quantiles(img, p_low, p_high):
       
    image = np.array(img)
    image = np.sort(image)
    x_low,x_high = np.percentile(image, [p_low, p_high]) 
    return x_low,x_high

def transform_by_lut(img, x_low, x_high):
    
    p1 = img < x_low
    p2 = img > x_high
    p3 = (img >= x_low) & (img <= x_high)
    p4 = (1/(x_high - x_low)) * (img - x_low)

    newImage = (p1.astype(float) * 0.0 + p2 * 1.0 + p3 * p4)
    
    fig = plt.figure(figsize= (40,6))
    plt.subplot(141)
    plt.imshow(img,cmap="gray");
    plt.title('image')
    plt.subplot(142)
    plt.imshow(newImage,cmap="gray");
    plt.title('Newimage')
    plt.subplot(143)
    plt.hist(img, bins=10)
    plt.title('image histogram')
    plt.subplot(144)
    plt.hist(newImage, bins=10)
    plt.title('newImage histogram');

# dont change this code, you can change P_LOW and P_HIGH of course
img = cv.imread(os.path.join("data", "Alexsandro_de_Souza.jpg"))
grayscaled = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(float) / 255.
P_LOW = 0.10
P_HIGH = 0.90
x_low, x_high = range_by_quantiles(grayscaled, P_LOW, P_HIGH)
transformed = transform_by_lut(grayscaled, x_low, x_high)

def advanced_lut(img, p_low, p_high, **kwargs):
      
    x_low, x_high = range_by_quantiles(img, p_low, p_high)

    p1 = img < x_low
    p2 = img > x_high
    p3 = (img >= x_low) & (img <= x_high)
    p4 = (1 / (x_high - x_low)) * (img - x_low)

    newImage = (p1.astype(float) * 0.0 + p2 * 1.0 + p3 * p4)

    return newImage


P_LOW = 0.01
P_HIGH = 0.99


transformed_image = advanced_lut(grayscaled, P_LOW, P_HIGH)


plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(grayscaled, cmap="gray")
plt.title('Original Image')
plt.subplot(122)
plt.imshow(transformed_image, cmap="gray")
plt.title('Transformed Image')
plt.show()

# dont change this code
img = cv.imread(os.path.join("data", "indir.png"))
grayscaled = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(float) / 255.
P_LOW = 0.01
P_HIGH = 0.99
transformed = advanced_lut(grayscaled, P_LOW, P_HIGH)