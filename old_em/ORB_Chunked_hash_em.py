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
img2_ = cv2.imread("enhanced/101_1.tif", cv2.IMREAD_GRAYSCALE)

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

# Compute key point distances
kp_dist = []
for match in matches:
    kp1_idx = match.queryIdx
    kp2_idx = match.trainIdx
    kp1_pos = kp1[kp1_idx].pt
    kp2_pos = kp2[kp2_idx].pt
    dist = np.sqrt((kp1_pos[0]-kp2_pos[0])**2 + (kp1_pos[1]-kp2_pos[1])**2)
    kp_dist.append(dist)

# Compute average key point distance
avg_kp_dist = sum(kp_dist)/len(kp_dist)

keypoint_match_score = np.exp(-np.array(kp_dist) / avg_kp_dist)

overall_score = np.average(keypoint_match_score)

print("-------overall_score -----------", overall_score )

# Define the threshold value for matching
match_threshold = 5

# Apply the ratio test to filter out false matches
matches = sorted(matches, key=lambda x: x.distance)
# The ratio test is applied to filter out false matches by sorting the matches by their distance and keeping only the matches that have a distance ratio lower than 0.75.
ratio = 0.75
num_good_matches = int(len(matches) * ratio)
matches = matches[:num_good_matches]

print("-----------------", num_good_matches)
# Check if the fingerprints match based on the number of good matches
if num_good_matches >= match_threshold:
    print("The fingerprints match!")
else:
    print("The fingerprints do not match.")


# Draw matches, Uses the keypoints and the matching information to draw lines between the corresponding points in the two images
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display image
cv2.imshow("Matches", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
