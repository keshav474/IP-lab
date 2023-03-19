# %% [markdown]
# ##                               IP LAB TERM PROJECT
# ### Identify curvilinear lines and separate them from straight lines in an image
# <hr>
# In this project, we aim to develop an algorithm that can distinguish between curved lines and straight lines in an image using edge detection methods followed by Harris corner detection. The goal of this project is to improve the accuracy of line segmentation in images and facilitate further analysis of the lines.
# 
# The project will not rely on machine learning methods but instead will focus on traditional image processing techniques to extract features and identify patterns. The approach will involve detecting edges in the image, which will be followed by the Harris corner detector to identify corners in the edges.
# 
# The output of the algorithm will be a binary image where the curved lines will be highlighted in one color and the straight lines in another color. 
# 
# Image is processed in the following manner:
# 
# 1. We start with grayscale image.
# 2. Apply sobel operator to highlight edges.
# 3. Use thresholding and edge thining to produce better image.
# 4. Apply Hough transform to detect lines.

# %%
import numpy as np
from math import pi
from numpy import cos, sin, rad2deg
from skimage import io
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel, sobel_h, sobel_v
from skimage.feature import canny,corner_harris, corner_peaks
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
# from matplotlib.pyplot import figure, imshow, title, subplot, plot, show, plot, xlabel, ylabel, axis
from matplotlib.pyplot import figure, imshow, title, subplot, plot, show, plot, xlabel, ylabel, axis, colorbar, clim,Circle




import time

# %% [markdown]
# #### Importing image from local and applying RGB to Grayscale Transform (only if image is RGB)
# <hr>

# %%
_img = io.imread('im.png')
# img = io.imread('proj4.png')
# img = io.imread('riv1.png')


figure(figsize=(20,17))

subplot(1,2,1)
title("Original image")
imshow(_img, cmap="gray")

if(len(_img.shape) == 3):
    img = rgb2gray(_img)
    img = img*255

subplot(1,2,2)
title("Original GrayScale image")
imshow(img, cmap="gray")

show()

# %% [markdown]
# ## Edge detection
# #### In this step we try to detect the edges of the image using the Sobel and Canny edge detector.
# 
# <hr>
# 

# %%
figure(figsize=(27,15))

subplot(2,3,1)
imshow(sobel_h(img), cmap="gray")
title("Horizontal gradients (sobel_h)")

subplot(2,3,2)
imshow(sobel_v(img), cmap="gray")
title("vertical gradients (sobel_v)")

subplot(2,3,3)
imshow(sobel(img), cmap="gray")
title("Gradients (sobel)")

figure(figsize=(24,17))
subplot(1,2,1)

temp = sobel(img)
for(i,j) in zip(*np.where(temp > 0.04)):
    temp[i][j] = 1

imshow(temp, cmap="gray")
title("Gradients (sobel) with thresholding")

show()


# %% [markdown]
# * NOTE :  An edge detection algorithm (e.g. sobel filter) does not actually "*detect*" edges, it rather "*strengthens*" regions with large changes (edges), and "*weakens*" regions with low changes. After that you can "*extract*" edges by applying some sort of threshold. The result would be a binary image
# <hr>

# %%
def binImg(img):
    img2 = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] > 14:
                img2[i,j] = 255
            else:
                img2[i,j] = 0
    return img2

# masking 3*3
def masking(_img, kernel):
    # wraping the image
    m, n = _img.shape
    img= np.zeros((m + 2, n + 2))
    img = img*255
    img[1:m + 1, 1:n + 1] = _img
    m,n = img.shape
    
    # op(img, cmap = 'gray')
    img2 = img.copy()
    
    for i in range(m - 2):
        for j in range(n - 2):
            img2[i + 1, j + 1] = np.sum(img[i:i + 3, j:j + 3] * kernel)
    # img2 = img2.astype(np.uint8)
    
    # imshow(img2, cmap = 'gray')
    img2[1, :] = 0
    img2[m - 2, :] = 0
    img2[:, 1] = 0
    img2[:, n - 2] = 0
    return img2[1:m - 1, 1:n - 1]
# masks
avg = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
laplacian = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
gaussian = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
prewittx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitty = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

def sobel_edge(img):
    return np.sqrt(masking(img, sobelx)**2 + masking(img, sobely)**2)


# %%
imgSobel = binImg(sobel(masking(img,avg))*255)

figure(figsize=(24,17))

subplot(1,2,1)
imgSobel = binImg(sobel_edge(masking(img,avg)))
title("Sovel edge detection using 3X3 mask")
imshow(imgSobel, cmap="gray")

subplot(1,2,2)
imshow(imgSobel, cmap="gray")
title("sobel with thresholding and binarization From skimage")


show()


# %% [markdown]
# > The edges formed by applying convolution kernels to an image are **thick** edges.<br>
# >> Our task of seperating detecting lines works far better with thinner edges.<br>
# >> So we use morphological  erosion to thin the edges.
# >  _________________________________________________________________
# 
# 
# 

# %%
def erosion(img):
    img2 = np.zeros_like(img)
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if img[i, j] == 255:
                if img[i-1, j-1] == 255 and img[i-1, j] == 255 and img[i-1, j+1] == 255 and img[i, j-1] == 255 and img[i, j+1] == 255 and img[i+1, j-1] == 255 and img[i+1, j] == 255 and img[i+1, j+1] == 255:
                    img2[i, j] = 255
    return img2

_imgSobel = erosion(imgSobel.copy())

figure(figsize=(17,14))
imshow(_imgSobel, cmap="gray")
title("Erosion")
show()

# %% [markdown]
# ### Harris corner detection

# %%
ks = [.05, 0.025, 0.2]

figure(figsize=(37,25))
for i, k in enumerate(ks):
    
    y = corner_harris(imgSobel, method='k', k=k, sigma=2)
    coords = corner_peaks(y, threshold_rel=.1)   
    subplot(1,3,i+1)
    imshow(img, cmap="gray")
    plot(coords[:,1], coords[:,0], color='lime', marker='+', linestyle='none')
    title(f"Result given by the function corner_peaks\nwith k = {k}")


# %% [markdown]
# ### Hough's Transform

# %%
# print(y)


# y = canny(masking(img, gaussian), sigma=2)
y = _imgSobel.copy()/255

accumulator, angle, dist  = hough_line(y)

peaks, angles, dists = hough_line_peaks(accumulator, angle, dist, threshold=.1*accumulator.max(), num_peaks=4)

# print(peaks, angles, dists)

axes = (rad2deg(angle[0]), rad2deg(angle[-1]), dist[-1], dist[0])

figure(figsize=(8,6))

imshow(accumulator**.5, cmap="gray", extent=axes, aspect="auto")
xlabel('Angle (degrees)')
ylabel('Distance (pixels)')
title("Hough transform")
plot(rad2deg(angles), dists, marker='o', markeredgecolor='lime', markerfacecolor="none", linestyle="none")

show()

M, N = img.shape

figure(figsize=(14,11))
imshow(y, cmap="gray")

for _, angle, dist in zip(peaks, angles, dists):
    
    x0 = 0
    y0 = dist / (sin(angle)+0.00000001)
    x1 = N
    y1 = (dist - x1*cos(angle)) / (sin(angle)+0.00000001)
    
    plot((x0, x1), (y0, y1), color='lime')

axis((0,N+10,M+10,0))
show()


# %% [markdown]
# > ### Probabilistic Hough Transform for straight line detection

# %%
# temp = (canny(masking(img, gaussian)))
temp = _imgSobel.copy()/255

N,M = temp.shape
lines = probabilistic_hough_line(temp, threshold=80, line_length=8, line_gap=3)
figure(figsize=(14,11))
imshow(_img)

for line in lines:
    p0, p1 = line
    plot((p0[0], p1[0]), (p0[1], p1[1]), color='r')

axis((0,M,N,0))
title("Straight Line detection")
show()

# %% [markdown]
# ## Curve detection 
# > In theorey Hough transform  can be used to detect any type of curves, but to detect any curve(other than straight lines) becomes computionally expensive and impractical.<br>
# > To detect curved objects in an image we will be using a counter intutive method. We will find all straight lines in the image and push them into the background using floodfill algorithm.<br><hr>

# %%
def floodfill12(img, x, y):
    if x>=0 and x<img.shape[0] and y>=0 and y<img.shape[1]:
        if img[x,y] == True:
            img[x,y] = False
        # print("for")
            if x+1 < img.shape[0] and y-1 >= 0:
                floodfill12(img, x+1, y-1)
            if x+1 < img.shape[0] and y+1 < img.shape[1]:
                floodfill12(img, x+1, y+1)
            if x-1 >= 0 and y-1 >= 0:
                floodfill12(img, x-1, y-1)
            if x-1 >= 0 and y+1 < img.shape[1]:
                floodfill12(img, x-1, y+1)
            if x+1 < img.shape[0]:
                floodfill12(img, x+1, y)
            if x-1 >= 0:
                floodfill12(img, x-1, y)
            if y+1 < img.shape[1]:
                floodfill12(img, x, y+1)
            if y-1 >= 0:
                floodfill12(img, x, y-1)
        
        # x = x
    return img

# %%
temp = (canny(masking(img, gaussian)))
# temp = _imgSobel.copy()/255
print(temp.shape)
# print(temp)

# temp[1][:]=1
for line in lines:
    p0, p1 = line
    print(p0, p1)
    M, N = temp.shape
    for i in range(p0[0]-3, p0[0]+3):
        for j in range(p0[1]-3, p0[1]+3):
            if i>=0 and i<N and j>=0 and j<M:
                if temp[j,i] == True:
                # print(j, i, temp[j,i])
                    floodfill12(temp, j, i)
                    break
    
    # for i in range(p1[0]-13, p1[0]+13):
    #     for j in range(p1[1]-3, p1[1]+3):
    #         if i>=0 and i<temp.shape[1] and j>=0 and j<temp.shape[0] and temp[j,i] == True:
    #             # print(j, i, temp[j,i])
    #             floodfill12(temp, j, i)

figure(figsize=(14,11))
# plot(i, j, color='red', marker='+', linestyle='none')
imshow(temp, cmap="gray")
# axis((0,M,N,0))
# floodfill12(temp, 80, 280, 0)
# floodfill12(temp, 89, 271, 0)
# for i in range(0, temp.shape[0]):
#     for j in range(0, temp.shape[1]):
#         if temp[i,j] == True:
#             # temp = floodfill12(temp, i, j, 0)
#             # plot(j, i, color='lime', marker='o', linestyle='none')
#             print(i, j, temp[i,j])

# title("Curves after removing the lines")



# %% [markdown]
# #### Hough Circle Transform

# %%
import matplotlib.pyplot as plt

from skimage.transform import hough_circle
from skimage.draw import circle_perimeter
from matplotlib.patches import Circle
# img = np.zeros((100, 100), dtype=bool)
# rr, cc = circle_perimeter(25, 35, 23)
# img[rr, cc] = 1
temp = sobel(img)
imshow(temp, cmap="gray")
for try_radii in range(70, 80):
    res = hough_circle(temp, try_radii)
    ridx, r, c = np.unravel_index(np.argmax(res), res.shape)
    # print(r, c, try_radii)
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img, cmap=plt.cm.gray)
    cr = Circle((c, r), try_radii,color = "r", fill=True, alpha=0.3)
    ax.add_patch(cr)
# (25, 35, 23)
# figure(figsize=(14,11))

for _, angle, dist in zip(peaks, angles, dists):
    # if abs(_) <= 75:
    #     continue
    x0 = 0
    y0 = dist / (sin(angle)+0.00001)
    x1 = N
    y1 = (dist - x1*cos(angle)) / (sin(angle)+0.00001)
    # print(y0,y1)
    plot((x0, x1), (y0, y1), color='lime')
axis((0,N,M,0))

show()



