import cv2
import numpy as np
import hashlib

# Load the fingerprint images
img1 = cv2.imread('enhanced/101_2.tif', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('enhanced/101_1.tif', cv2.IMREAD_GRAYSCALE)

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

# Divide concatenated hash values into blocks
hash_block_size = 64  # Number of characters per block
hash1_blocks = [hash1[i:i+hash_block_size] for i in range(0, len(hash1), hash_block_size)]
hash2_blocks = [hash2[i:i+hash_block_size] for i in range(0, len(hash2), hash_block_size)]

# Create a BFMatcher object
bf = cv2.BFMatcher()

# Compare blocks using the BFMatcher and ratio test
matches = []

similarity_score = 0
for i in range(len(des1_blocks)):
    des1_block = des1_blocks[i]
    if i < len(des2_blocks):
        des2_block = des2_blocks[i]
        matches = bf.knnMatch(des1_block, des2_block, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)
        similarity_score += len(good_matches)

print('------------', similarity_score)



# for i in range(len(des1_blocks)):
#     des1_block = des1_blocks[i]
#     des2_block = des2_blocks[i]
    
#     # Match the descriptors using the BFMatcher
#     block_matches = bf.knnMatch(des1_block, des2_block, k=2)
    
#     # Apply ratio test to filter good matches
#     good_block_matches = []
#     for m,n in block_matches:
#         if m.distance < 0.75*n.distance:
#             good_block_matches.append(m)
            
#     matches.append(len(good_block_matches))

# total_descriptors = len(des1) + len(des2)
# total_good_matches = sum(matches)
# match_score = total_good_matches / total_descriptors