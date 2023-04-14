import cv2
import numpy as np
import hashlib
import enhance_function

# Load the fingerprint images
img1_ = cv2.imread('enhanced/101_1.tif', cv2.IMREAD_GRAYSCALE)
img2_ = cv2.imread('enhanced/101_1.tif', cv2.IMREAD_GRAYSCALE)

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

# print('------hash------',hash1)

# -------------------------------------

# Divide concatenated hash values into blocks
hash_block_size = len(des1)//block_size  # Number of characters per block
hash1_blocks_ = [hash1[i:i+hash_block_size] for i in range(0, len(hash1), hash_block_size)]
hash2_blocks_ = [hash2[i:i+hash_block_size] for i in range(0, len(hash2), hash_block_size)]

# Convert hash blocks into numerical values
# Convert hash blocks to numpy arrays of floats
hash1_blocks_ = np.array([np.frombuffer(bytes.fromhex(block), dtype=np.float32) for block in hash1_blocks])
hash2_blocks_ = np.array([np.frombuffer(bytes.fromhex(block), dtype=np.float32) for block in hash2_blocks])


# # Create FLANN-based matcher object
# index_params = dict(algorithm=0, trees=5)
# search_params = dict(checks=50)
# flann = cv2.FlannBasedMatcher(index_params, search_params)

# # Build FLANN index for array1
# index_params = dict(algorithm=1, trees=5)
# index = cv2.flann_Index(np.float32(hash1_floats), index_params)

# # Search for nearest neighbors in array2
# distances, indices = index.knnSearch(np.float32(hash2_floats), k=1, params={})

# # Calculate similarity score
# similarity_score = np.mean(distances)
# print('Similarity score:', similarity_score)





# Create FLANN index and parameters
index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Compare blocks using the FLANN index and ratio test
similarity_score = 0
for i in range(len(hash1_blocks_)):
    des1_hash_block = hash1_blocks_[i]
    if i < len(hash2_blocks_) and len(des1_hash_block) == len(des2_blocks):
        des2_block = hash2_blocks_[i]
        matches = flann.knnMatch(des1_hash_block, des2_block, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)
        similarity_score += len(good_matches)

print('Similarity score:', similarity_score)