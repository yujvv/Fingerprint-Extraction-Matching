import cv2
import numpy as np
import random
import enhance_function
import hash

# adjust the hashSize, numHashes, and ratio values to optimize the matching performance

# Initialize feature detector and descriptor
detector = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Load images
# enhanced/1__M_Left_index_finger_CR.BMP
# fingerprint_c/1.jpg
img1_ = cv2.imread("enhanced/102_3.tif", cv2.IMREAD_GRAYSCALE)
img2_ = cv2.imread("enhanced/104_2.tif", cv2.IMREAD_GRAYSCALE)

img1 = enhance_function.enhance_fingerprint_2(img1_)
img2 = enhance_function.enhance_fingerprint_2(img2_)

# Detect keypoints and compute descriptors, the second parameter is a mask to limit the region
# kp is the keypoints detected in the image
# des is the descriptors computed for each keypoint (2D array of floats) (vector of numbers)
kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)

# Hash the descriptors, size of blocks, numHashes random hash functions
blockSize = 4
hashSize = 16
# Each hash function selects hashSize blocks from the descriptors
numHashes = 4

# Can be compared by Hamming distance or Jaccard similarity
enc_des1 = hash.hash_descriptors_2(des1, blockSize, hashSize, numHashes)
enc_des2 = hash.hash_descriptors_2(des2, blockSize, hashSize, numHashes)

# Convert float data type to CV_8U (unsigned integer type with 8 bits) (ensure data is within the range of 0-255)
enc_des1 = cv2.convertScaleAbs(enc_des1)
enc_des2 = cv2.convertScaleAbs(enc_des2)

enc_des1 = np.array(enc_des1).astype(np.uint8)
enc_des2 = np.array(enc_des2).astype(np.uint8)

# Match descriptors, find the best matching features between the two sets of descriptors, Brute-Force Matcher (BFMatcher) or Fast Approximate Nearest Neighbor (FLANN)
matches = matcher.match(enc_des1, enc_des2)

# Compute key point distances and strengths
kp_dist = []
kp_strengths = []
for match in matches:
    kp1_idx = match.queryIdx
    kp2_idx = match.trainIdx
    kp1_pos = kp1[kp1_idx].pt
    kp2_pos = kp2[kp2_idx].pt
    dist = np.sqrt((kp1_pos[0]-kp2_pos[0])**2 + (kp1_pos[1]-kp2_pos[1])**2)
    strength1 = kp1[kp1_idx].response
    strength2 = kp2[kp2_idx].response
    kp_dist.append(dist)
    kp_strengths.append(strength1 * strength2)

# Compute average key point distance and strength
avg_kp_dist = sum(kp_dist)/len(kp_dist)
avg_kp_strength = sum(kp_strengths)/len(kp_strengths)

# Weight key point match scores by strength
keypoint_match_score = np.exp(-np.array(kp_dist) / avg_kp_dist) * np.array(kp_strengths) / avg_kp_strength

# Compute overall score
overall_score = np.average(keypoint_match_score)

# Define the threshold value for matching
match_threshold = 36

# Apply the ratio test to filter out false matches
matches = sorted(matches, key=lambda x: x.distance)
good_matches = []
for match in matches:
    kp1_idx = match.queryIdx
    kp2_idx = match.trainIdx
    kp1_pos = kp1[kp1_idx].pt
    kp2_pos = kp2[kp2_idx].pt
    dist = np.sqrt((kp1_pos[0]-kp2_pos[0])**2 + (kp1_pos[1]-kp2_pos[1])**2)
    strength1 = kp1[kp1_idx].response
    strength2 = kp2[kp2_idx].response
    score = np.exp(-dist / avg_kp_dist) * (strength1 * strength2) / avg_kp_strength
    if score >= 0.7 * overall_score:
        good_matches.append(match)

num_good_matches = len(good_matches)

print("-----------------", num_good_matches)

# Check if the fingerprints match based on the number of good matches
if num_good_matches >= match_threshold:
    print("The fingerprints match!")
else:
    print("The fingerprints do not match.")
