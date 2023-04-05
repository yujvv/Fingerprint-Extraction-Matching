import cv2
import numpy as np
import hashlib
import enhance_function

# Load the fingerprint images
img1_ = cv2.imread('enhanced/101_2.tif', cv2.IMREAD_GRAYSCALE)
img2_ = cv2.imread('enhanced/103_1.tif', cv2.IMREAD_GRAYSCALE)

# a and n are both ok
img1 = enhance_function.enhance_fingerprint_a(img1_)
img2 = enhance_function.enhance_fingerprint_a(img2_)

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

print('------hash------',hash1)

# -------------------------------------

# Divide concatenated hash values into blocks
hash_block_size = 64  # Number of characters per block
hash1_blocks = [hash1[i:i+hash_block_size] for i in range(0, len(hash1), hash_block_size)]
hash2_blocks = [hash2[i:i+hash_block_size] for i in range(0, len(hash2), hash_block_size)]


# Create a FLANN object
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Build the FLANN index from the descriptors of the second image
des2_float = np.float32(des2)
flann.add([des2_float])

# Compare blocks using the FLANN index and ratio test
similarity_score = 0
for i in range(len(des1_blocks)):
    des1_block = des1_blocks[i]
    if i < len(des2_blocks):
        des2_block = des2_blocks[i]
        matches = flann.knnMatch(des1_block, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)
        similarity_score += len(good_matches)

print('Similarity score:', similarity_score)

# print('----', len(des1_blocks) == len(des2_blocks))