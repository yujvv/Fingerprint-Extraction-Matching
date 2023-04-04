import cv2
import hashlib
import enhance_function
import Levenshtein

# Define the block size
BLOCK_SIZE = 16

# Load the fingerprint images
img1_ = cv2.imread('enhanced/103_3.tif', cv2.IMREAD_GRAYSCALE)
img2_ = cv2.imread('enhanced/102_1.tif', cv2.IMREAD_GRAYSCALE)

img1 = enhance_function.enhance_fingerprint_2(img1_)
img2 = enhance_function.enhance_fingerprint_2(img2_)

# Create the SIFT detector object
sift = cv2.SIFT_create()

# Split the images into blocks and extract SIFT descriptors for each block
des1_blocks = []
des2_blocks = []
for i in range(0, img1.shape[0], BLOCK_SIZE):
    for j in range(0, img1.shape[1], BLOCK_SIZE):
        block1 = img1[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
        block2 = img2[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
        kp1, des1 = sift.detectAndCompute(block1, None)
        kp2, des2 = sift.detectAndCompute(block2, None)
        if des1 is not None:
            des1_blocks.append(des1)
        if des2 is not None:
            des2_blocks.append(des2)

# Hash the SIFT descriptors using SHA-256
hash1_blocks = [hashlib.sha256(des.tostring()).digest() for des in des1_blocks]
hash2_blocks = [hashlib.sha256(des.tostring()).digest() for des in des2_blocks]

# Combine the hashes for all the blocks
combined_hash1 = hashlib.sha256(b''.join(hash1_blocks)).hexdigest()
combined_hash2 = hashlib.sha256(b''.join(hash2_blocks)).hexdigest()


# print('---------------', combined_hash1)
# print('---------------', combined_hash2)


# hamming_distance = 0
# for i in range(len(combined_hash1)):
#     if combined_hash1[i] != combined_hash2[i]:
#         hamming_distance += 1
# similarity = (len(combined_hash1) - hamming_distance) / len(combined_hash1)

# print('--------------------', similarity)

lev_distance = Levenshtein.distance(combined_hash1, combined_hash2)
match_score = 1 - (lev_distance / max(len(combined_hash1), len(combined_hash2)))
print("-------", match_score)


# # Create a BFMatcher object
# bf = cv2.BFMatcher()

# # Match the descriptors using the BFMatcher
# matches = bf.knnMatch(combined_hash1, combined_hash2, k=2)

# # Apply ratio test to filter good matches
# good_matches = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good_matches.append(m)

# print('---------------', len(good_matches))
