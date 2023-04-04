import cv2
import numpy as np
import random
import enhance_function
import hash

# Load the fingerprint images
img1 = cv2.imread('enhanced/103_3.tif', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('enhanced/101_1.tif', cv2.IMREAD_GRAYSCALE)

# Create the SIFT detector object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


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


# Create a BFMatcher object
bf = cv2.BFMatcher()

# Match the descriptors using the BFMatcher
matches = bf.knnMatch(enc_des1, enc_des2, k=2)

# Apply ratio test to filter good matches
good_matches = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_matches.append(m)

print('---------------', len(good_matches))

# Draw the matches on a new image
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)

# Display the image
cv2.imshow('Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
