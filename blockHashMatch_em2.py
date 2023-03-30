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

# Detect keypoints and compute descriptors, the second parameter is a mask to limit the region
# kp is the keypoints detected in the image
# des is the descriptors computed for each keypoint (2D array of floats) (vector of numbers)
kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)

# Hash the descriptors, size of blocks, numHashes random hash functions
blockSize = 4
hashSize = 8
# Each hash function selects hashSize blocks from the descriptors
numHashes = 2

# Performs feature hashing on the descriptors, map high-dimensional feature vectors to a lower-dimensional space 
def hash_descriptors(descriptors, blockSize, hashSize, numHashes):
    hashList = []
    for i in range(numHashes):
        random.seed(i)
        hashFunc = random.sample(range(descriptors.shape[1]), hashSize)
        hashVal = []
        # For each descriptor, the selected blocks for each hash function are compared to their mean values, and a binary hash is generated based on whether each block is above or below its mean
        for descriptor in descriptors:
            hash = ""
            for j in hashFunc:
                block = descriptor[j*blockSize:(j+1)*blockSize]
                mean = np.mean(block)
                hash += "1" if (block > mean).sum() >= 2 else "0"
            hashVal.append(int(hash, 2))
        hashList.append(np.array(hashVal))
    # The hash values for all descriptors are stacked vertically to produce the final hash matrix (numDescriptors, numHashes)
    return np.vstack(hashList).T

# Can be compared by Hamming distance or Jaccard similarity
enc_des1 = hash_descriptors(des1, blockSize, hashSize, numHashes)
enc_des2 = hash_descriptors(des2, blockSize, hashSize, numHashes)

# Convert float data type to CV_8U (unsigned integer type with 8 bits) (ensure data is within the range of 0-255)
enc_des1 = cv2.convertScaleAbs(enc_des1)
enc_des2 = cv2.convertScaleAbs(enc_des2)

enc_des1 = np.array(enc_des1).astype(np.uint8)
enc_des2 = np.array(enc_des2).astype(np.uint8)

# Match descriptors, find the best matching features between the two sets of descriptors, Brute-Force Matcher (BFMatcher) or Fast Approximate Nearest Neighbor (FLANN)
matches = matcher.match(enc_des1, enc_des2)

print('---------start----------')
print("matches-----",matches)
print('---------end----------')
# # Apply the ratio test to filter out false matches
# matches = sorted(matches, key=lambda x: x.distance)
# ratio = 0.75
# num_good_matches = int(len(matches) * ratio)
# matches = matches[:num_good_matches]

# # Draw matches
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# # Display image
# cv2.imshow("Matches", img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# Define the threshold value for matching
match_threshold = 10

# Apply the ratio test to filter out false matches
matches = sorted(matches, key=lambda x: x.distance)
# The ratio test is applied to filter out false matches by sorting the matches by their distance and keeping only the matches that have a distance ratio lower than 0.75.
ratio = 0.75
num_good_matches = int(len(matches) * ratio)
matches = matches[:num_good_matches]

# Draw matches, Uses the keypoints and the matching information to draw lines between the corresponding points in the two images
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Check if the fingerprints match based on the number of good matches
if num_good_matches >= match_threshold:
    print("The fingerprints match!")
else:
    print("The fingerprints do not match.")

# Display image
cv2.imshow("Matches", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
