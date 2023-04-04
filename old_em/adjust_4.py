import cv2
import numpy as np
import hashlib
import Levenshtein

# Load the fingerprint images
img1 = cv2.imread('enhanced/103_3.tif', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('enhanced/101_3.tif', cv2.IMREAD_GRAYSCALE)

# Create the SIFT detector object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Divide descriptors into blocks
block_size = 64  # Number of descriptors per block
des1_blocks = np.array_split(des1, len(des1) // block_size)
des2_blocks = np.array_split(des2, len(des2) // block_size)

# Apply hashing to each block and concatenate hashed values
hash1_blocks = [hashlib.sha256(block.tobytes()).hexdigest() for block in des1_blocks]
hash2_blocks = [hashlib.sha256(block.tobytes()).hexdigest() for block in des2_blocks]
hash1 = ''.join(hash1_blocks)
hash2 = ''.join(hash2_blocks)


# hamming_distance = 0
# for i in range(len(hash2)):
#     if hash1[i] != hash2[i]:
#         hamming_distance += 1
# similarity = (len(hash2) - hamming_distance) / len(hash2)

# print('--------------------', similarity)

lev_distance = Levenshtein.distance(hash1, hash2)
match_score = 1 - (lev_distance / max(len(hash1), len(hash2)))
print("-------", match_score)