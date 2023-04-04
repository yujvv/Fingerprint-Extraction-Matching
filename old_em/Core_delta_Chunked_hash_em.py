import cv2
import numpy as np
import random

# adjust the hashSize, numHashes, and ratio values to optimize the matching performance

# Initialize feature detector and descriptor
detector = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Load images
img1 = cv2.imread("enhanced/2.jpg", 0)
img2 = cv2.imread("enhanced/3.jpg", 0)

# Define the Core and Delta Regions
core_region = (160, 160, 160, 160)  # x, y, w, h
delta_region = (0, 320, 320, 160)  # x, y, w, h

# Define the block size
block_size = 16

# Detect keypoints and compute descriptors for the Core Region
core_img1 = img1[core_region[1]:core_region[1]+core_region[3], core_region[0]:core_region[0]+core_region[2]]
core_img2 = img2[core_region[1]:core_region[1]+core_region[3], core_region[0]:core_region[0]+core_region[2]]

kp1_core, des1_core = detector.detectAndCompute(core_img1, None)
kp2_core, des2_core = detector.detectAndCompute(core_img2, None)


# Performs feature hashing on the descriptors, map high-dimensional feature vectors to a lower-dimensional space 

def hash_descriptors(descriptors, block_size):
    """
    Hashes a set of descriptors into a set of blocks.
    Args:
        descriptors (np.ndarray): An array of shape (n_descriptors, desc_size, desc_size) representing the
            descriptors to be hashed.
        block_size (int): The size of the blocks in which the descriptors will be hashed.
    Returns:
        np.ndarray: An array of shape (n_blocks, block_size, block_size) representing the hashed blocks.
    """
    n_descriptors, desc_size = descriptors.shape
    n_blocks_per_dim = desc_size // block_size
    n_blocks = n_descriptors * n_blocks_per_dim ** 2

    # Initialize the output array with zeros.
    hashed_descriptors = np.zeros((n_blocks, block_size, block_size))

    # Iterate over each descriptor and hash it into blocks.
    for i in range(n_descriptors):
        descriptor = descriptors[i]

        for j in range(n_blocks_per_dim):
            for k in range(n_blocks_per_dim):
                block = descriptor[j * block_size * desc_size + i * block_size : j * block_size * desc_size + (i + 1) * block_size]

                index = i * n_blocks_per_dim ** 2 + j * n_blocks_per_dim + k
                hashed_descriptors[index] = block

    return hashed_descriptors


# Hash the descriptors for the Core Region
enc_des1_core = hash_descriptors(des1_core, block_size)
enc_des2_core = hash_descriptors(des2_core, block_size)

# Detect keypoints and compute descriptors for the Delta Region
delta_img1 = img1[delta_region[1]:delta_region[1]+delta_region[3], delta_region[0]:delta_region[0]+delta_region[2]]
delta_img2 = img2[delta_region[1]:delta_region[1]+delta_region[3], delta_region[0]:delta_region[0]+delta_region[2]]

kp1_delta, des1_delta = detector.detectAndCompute(delta_img1, None)
kp2_delta, des2_delta = detector.detectAndCompute(delta_img2, None)

# Hash the descriptors for the Delta Region
enc_des1_delta = hash_descriptors(des1_delta, block_size)
enc_des2_delta = hash_descriptors(des2_delta, block_size)

# Concatenate the hashes for the Core and Delta Regions
enc_des1 = np.hstack((enc_des1_core, enc_des1_delta))
enc_des2 = np.hstack((enc_des2_core, enc_des2_delta))

# Match descriptors, find the best matching features between the two sets of descriptors, Brute-Force Matcher (BFMatcher) or Fast Approximate Nearest Neighbor (FLANN)
matches = matcher.match(enc_des1, enc_des2)

# Define the threshold value for matching
match_threshold = 10

# Apply the ratio test to filter out false matches
matches = sorted(matches, key=lambda x: x.distance)
ratio = 0.75
num_good_matches = int(len(matches) * ratio)
matches = matches[:num_good_matches]

# Check if the fingerprints match based on the number of good matches
if num_good_matches >= match_threshold:
    print("The fingerprints match!")
else:
    print("The fingerprints do not match.")

