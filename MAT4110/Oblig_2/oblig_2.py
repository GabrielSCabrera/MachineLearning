from imageio import imread, imwrite
import matplotlib.pyplot as plt
import numpy as np

# Convert the images to grayscale
# https://pixabay.com/en/board-chess-chessboard-black-white-157165/
# https://pixabay.com/en/jellyfish-under-water-sea-ocean-698521/
# https://pixabay.com/en/new-york-city-skyline-nyc-690868/

im1 = np.mean(imread("new_york.jpg"), axis = -1)
im2 = np.mean(imread("jellyfish.jpg"), axis = -1)
im3 = np.mean(imread("chessboard.png"), axis = -1)

# Convert to uint8 for compatibility
im1 = np.round(im1).astype(np.uint8)
im2 = np.round(im2).astype(np.uint8)
im3 = np.round(im3).astype(np.uint8)

# Plot the images
imwrite("New_York.png", im1)
imwrite("Manet.png", im2)
imwrite("Sjakkbrett.png", im3)

# Performing the SVD
U1, D1, VT1 = np.linalg.svd(im1) # (720, 695)
U2, D2, VT2 = np.linalg.svd(im2) # (640, 960)
U3, D3, VT3 = np.linalg.svd(im3) # (640, 960)

# Dimensions
# U1, D1, VT1: (640, 640) (640,) (960, 960)
# U2, D2, VT2: (640, 640) (640,) (960, 960)
# U3, D3, VT3: (720, 720) (695,) (695, 695)

# Calculating the uncompressed sum of each matrix' size
S1U = im1.size
S2U = im2.size
S3U = im3.size

dn1, dn2, dn3 = 330, 500, 718
dm1, dm2, dm3 = 700, 700, 691

# Size of N
n1 = 640 - dn1
n2 = 640 - dn2
n3 = 720 - dn3

# Size of M
m1 = 960 - dm1
m2 = 960 - dm2
m3 = 695 - dm3

# Retained singular values – maximum values are (r1 = 695; r2 = r3 = 960)
r1 = n1
r2 = n2
r3 = n3

# Plotting the singular values
plt.semilogy(np.arange(1, len(D1)+1), D1)
plt.xlim([1, len(D1)])
plt.xlabel("Singular Values $\\sigma_i$")
plt.ylabel("Magnitude")
plt.savefig("singvals_0.pdf", dpi = 250)
plt.close()

plt.semilogy(np.arange(1, len(D2)+1), D2)
plt.xlim([1, len(D2)])
plt.xlabel("Singular Values $\\sigma_i$")
plt.ylabel("Magnitude")
plt.savefig("singvals_1.pdf", dpi = 250)
plt.close()

plt.semilogy(np.arange(1, len(D3)+1), D3)
plt.xlim([1, len(D3)])
plt.xlabel("Singular Values $\\sigma_i$")
plt.ylabel("Magnitude")
plt.savefig("singvals_2.pdf", dpi = 250)
plt.close()

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
imwrite("New_York_compressed.png", im1_compressed)
imwrite("Manet_compressed.png", im2_compressed)
imwrite("Sjakkbret_compressed.png", im3_compressed)

# Calculating the compression ratios
R1 = S1U/S1C
R2 = S2U/S2C
R3 = S3U/S3C
msg = (f"\nCOMPRESSION RATES\n\tNew York:\t{R1:.2f}\n\tJellyfish:\t{R2:.2f}"
       f"\n\tChessboard:\t{R3:.2f}")
print(msg)
