from pathlib import Path
from functions import *


# Parameters
blob_ratio = 0.01
r = 5
eps = 0.3
w = 7


input_folder = Path('examples')
path_A = input_folder / 'lytro-05-A.jpg'
path_B = input_folder / 'lytro-05-B.jpg'

A = cv2.imread(path_A.as_posix())
B = cv2.imread(path_B.as_posix())

A_gray = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
A_gray = A_gray.astype(float) / 255.0

B_gray = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
B_gray = B_gray.astype(float) / 255.0

height, width = A_gray.shape
area = np.ceil(blob_ratio * height * width)

# Mean filtered
mf_A = mean_filter(A_gray, kernel_size=w)
mf_B = mean_filter(B_gray, kernel_size=w)

# Rough focus maps
rfm_A = np.abs(A_gray - mf_A)
rfm_B = np.abs(B_gray - mf_B)

# First guided focus maps
gfm_1_A = guidedfilter(A_gray, rfm_A, r, eps)
gfm_1_B = guidedfilter(B_gray, rfm_B, r, eps)

# Initial decision map
idm = gfm_1_A > gfm_1_B
idm = remove_blobs(idm, area)

mmap = np.multiply(idm.astype(float), A_gray) + np.multiply((1 - idm), B_gray)

# Final decision map
fdm = guidedfilter(mmap, idm.astype(np.uint8), r, eps)

##

if A.shape[2] > 1:
    # idm = np.repeat(idm[:, :, np.newaxis], 3, axis=2)
    fdm = np.repeat(fdm[:, :, np.newaxis], 3, axis=2)

# fusion_GF1 = np.multiply(idm, A) + np.multiply((1 - idm), B)
# fusion_GF1 = fusion_GF1.astype(np.uint8)

fusion_GF2 = np.multiply(fdm, A) + np.multiply((1 - fdm), B)
fusion_GF2 = fusion_GF2.astype(np.uint8)

