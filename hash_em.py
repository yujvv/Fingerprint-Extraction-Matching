import fingerprint_feature_extractor
import cv2
import numpy as np

# Read the fingerprint images
img1 = cv2.imread('enhanced/1.jpg', 0)
img2 = cv2.imread('enhanced/3.jpg', 0)

# Extract minutiae features from both fingerprints
FeaturesTerminations1, FeaturesBifurcations1 = fingerprint_feature_extractor.extract_minutiae_features(img1, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False)
FeaturesTerminations2, FeaturesBifurcations2 = fingerprint_feature_extractor.extract_minutiae_features(img2, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False)

# Combine both types of features for each fingerprint into a single list
Features1 = FeaturesTerminations1 + FeaturesBifurcations1
Features2 = FeaturesTerminations2 + FeaturesBifurcations2

# Create keypoint objects for each feature in both fingerprints
# kp1 = [cv2.KeyPoint(x=f.locX, y=f.locY, _size=10) for f in Features1]
# kp2 = [cv2.KeyPoint(x=f.locX, y=f.locY, _size=10) for f in Features2]
kp1 = [cv2.KeyPoint(x=f.locX, y=f.locY, size=10) for f in Features1]
kp2 = [cv2.KeyPoint(x=f.locX, y=f.locY, size=10) for f in Features2]



# Create a SIFT descriptor object
sift = cv2.SIFT_create()

# Calculate SIFT descriptors for each keypoint in both fingerprints
_, des1 = sift.compute(img1, kp1)
_, des2 = sift.compute(img2, kp2)

# Create a BFMatcher object to match SIFT descriptors
bf = cv2.BFMatcher()

# Match SIFT descriptors of both fingerprints using the BFMatcher object
matches = bf.match(des1, des2)

# Sort the matches in ascending order of their distances
matches = sorted(matches, key=lambda x: x.distance)

# Keep only the top 10% of matches based on distance
matches = matches[:int(len(matches) * 0.1)]

if len(matches) > 0:
    print("Fingerprints matched!")
else:
    print("Fingerprints do not match.")
