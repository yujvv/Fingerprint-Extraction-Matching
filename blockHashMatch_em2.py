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

# Detect keypoints and compute descriptors
kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)

# Hash the descriptors
blockSize = 4
hashSize = 8
numHashes = 2

def hash_descriptors(descriptors, blockSize, hashSize, numHashes):
    hashList = []
    for i in range(numHashes):
        random.seed(i)
        hashFunc = random.sample(range(descriptors.shape[1]), hashSize)
        hashVal = []
        for descriptor in descriptors:
            hash = ""
            for j in hashFunc:
                block = descriptor[j*blockSize:(j+1)*blockSize]
                mean = np.mean(block)
                hash += "1" if (block > mean).sum() >= 2 else "0"
            hashVal.append(int(hash, 2))
        hashList.append(np.array(hashVal))
    return np.vstack(hashList).T

enc_des1 = hash_descriptors(des1, blockSize, hashSize, numHashes)
enc_des2 = hash_descriptors(des2, blockSize, hashSize, numHashes)

# Convert data type to CV_8U
enc_des1 = cv2.convertScaleAbs(enc_des1)
enc_des2 = cv2.convertScaleAbs(enc_des2)

enc_des1 = np.array(enc_des1).astype(np.uint8)
enc_des2 = np.array(enc_des2).astype(np.uint8)

# Match descriptors
matches = matcher.match(enc_des1, enc_des2)

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
ratio = 0.75
num_good_matches = int(len(matches) * ratio)
matches = matches[:num_good_matches]

# Draw matches
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
