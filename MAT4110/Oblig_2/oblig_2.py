from imageio import imread, imwrite
import matplotlib.pyplot as plt
import numpy as np

# Convert the images to grayscale
# https://pixabay.com/en/board-chess-chessboard-black-white-157165/
# https://pixabay.com/en/jellyfish-under-water-sea-ocean-698521/
# https://pixabay.com/en/new-york-city-skyline-nyc-690868/

im1 = np.mean(imread("chessboard.png"), axis = -1)
im2 = np.mean(imread("jellyfish.jpg"), axis = -1)
im3 = np.mean(imread("new_york.jpg"), axis = -1)

# Convert to uint8 for compatibility
im1 = np.round(im1).astype(np.uint8)
im2 = np.round(im2).astype(np.uint8)
im3 = np.round(im3).astype(np.uint8)

# Plot the images
imwrite("Sjakkbrett.png", im1)
imwrite("Manet.png", im2)
imwrite("New York.png", im3)

# Performing the SVD
U1, D1, VT1 = np.linalg.svd(im1)#, full_matrices = False)
U2, D2, VT2 = np.linalg.svd(im2)#, full_matrices = False)
U3, D3, VT3 = np.linalg.svd(im3)#, full_matrices = False)

# Dimensions
# U1, D1, VT1: (720, 720) (695,) (695, 695)
# U2, D2, VT2: (640, 640) (640,) (960, 960)
# U3, D3, VT3: (640, 640) (640,) (960, 960)

# Calculating the uncompressed sum of each matrix' size
S1U = U1.size + D1.size + VT1.size
S2U = U2.size + D2.size + VT2.size
S3U = U3.size + D3.size + VT3.size

# Retained singular values – maximum values are (r1 = 695; r2 = r3 = 960)
r1 = 69
r2 = 96
r3 = 96

# Size of N
n1 = 72
n2 = 64
n3 = 64

# Size of M
m1 = 69
m2 = 96
m3 = 96

# Shrinking the singular value arrays (compressing the image)
D1 = D1[:r1]
D2 = D2[:r2]
D3 = D3[:r3]

# Resizing U
U1 = U1[:, :n1]
U2 = U2[:, :n2]
U3 = U3[:, :n3]

# Resizing VT
VT1 = VT1[:m1, :]
VT2 = VT2[:m2, :]
VT3 = VT3[:m3, :]

# Calculating the compressed sum of each matrix' size
S1C = U1.size + D1.size + VT1.size
S2C = U2.size + D2.size + VT2.size
S3C = U3.size + D3.size + VT3.size

# Creating the S matrices
S1 = np.zeros((U1.shape[1], VT1.shape[0]))
np.fill_diagonal(S1, D1)

S2 = np.zeros((U2.shape[1], VT2.shape[0]))
np.fill_diagonal(S2, D2)

S3 = np.zeros((U3.shape[1], VT3.shape[0]))
np.fill_diagonal(S3, D3)

# Decompressing the image
im1_compressed = np.round(U1 @ S1 @ VT1)
im2_compressed = np.round(U2 @ S2 @ VT2)
im3_compressed = np.round(U3 @ S3 @ VT3)

# Converting to uint8
im1_compressed = im1_compressed.astype(np.uint8)
im2_compressed = im2_compressed.astype(np.uint8)
im3_compressed = im3_compressed.astype(np.uint8)

# Saving the decompressed image
imwrite("Sjakkbret_compressed.png", im1_compressed)
imwrite("Manet_compressed.png", im2_compressed)
imwrite("New York_compressed.png", im3_compressed)

# Calculating the compression ratios
R1 = S1C/S1U
R2 = S2C/S2U
R3 = S3C/S3U
msg = (f"\nCOMPRESSION RATES\n\tChessboard:\t{R1:.2f}\n\tJellyfish:\t{R2:.2f}"
       f"\n\tNew York:\t{R3:.2f}")
print(msg)
