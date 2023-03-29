import fingerprint_feature_extractor
import cv2
import numpy as np

# Read the fingerprint images
img1 = cv2.imread('enhanced/1.jpg', 0)
img2 = cv2.imread('enhanced/3.jpg', 0)

# Extract minutiae features from both fingerprints
# spuriousMinutiaeThresh: This parameter controls the threshold value for filtering out spurious minutiae.
FeaturesTerminations1, FeaturesBifurcations1 = fingerprint_feature_extractor.extract_minutiae_features(img1, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False)
FeaturesTerminations2, FeaturesBifurcations2 = fingerprint_feature_extractor.extract_minutiae_features(img2, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False)

# Combine both types of features for each fingerprint into a single list
Features1 = FeaturesTerminations1 + FeaturesBifurcations1
Features2 = FeaturesTerminations2 + FeaturesBifurcations2

# Create keypoint objects for each feature in both fingerprints
kp1 = [cv2.KeyPoint(x=f.locX, y=f.locY, size=10) for f in Features1]
kp2 = [cv2.KeyPoint(x=f.locX, y=f.locY, size=10) for f in Features2]

# Create a SIFT descriptor object
sift = cv2.SIFT_create()

# Calculate the SIFT descriptors for the keypoints in both fingerprints
kp1, des1 = sift.compute(img1, kp1)
kp2, des2 = sift.compute(img2, kp2)

# Create a BFMatcher object and match the SIFT descriptors of the two fingerprints
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply the ratio test to filter out false matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Determine if the fingerprints match based on the number of good matches
if len(good_matches) > 10:
    print("Fingerprints matched!")
else:
    print("Fingerprints do not match.")

# Draw the good matches on a new image and display it
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Good Matches", img_matches)
cv2.waitKey(0)
