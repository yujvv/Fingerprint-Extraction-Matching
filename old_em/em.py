import fingerprint_feature_extractor
import cv2
import numpy as np

img1 = cv2.imread('enhanced/2.jpg', 0)
img2 = cv2.imread('enhanced/1.jpg', 0)

# Extracting minutiae features
FeaturesTerminations1, FeaturesBifurcations1 = fingerprint_feature_extractor.extract_minutiae_features(img1, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False)
FeaturesTerminations2, FeaturesBifurcations2 = fingerprint_feature_extractor.extract_minutiae_features(img2, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False)

# Matching minutiae features
matched_features = []
for t1 in FeaturesTerminations1:
    for t2 in FeaturesTerminations2:
        if abs(t1.locX - t2.locX) < 5 and abs(t1.locY - t2.locY) < 5:
            matched_features.append((t1.locX, t1.locY, t2.locX, t2.locY))
            break

for b1 in FeaturesBifurcations1:
    for b2 in FeaturesBifurcations2:
        if abs(b1.locX - b2.locX) < 5 and abs(b1.locY - b2.locY) < 5:
            matched_features.append((b1.locX, b1.locY, b2.locX, b2.locY))
            break

if len(matched_features) > 0:
    print("Fingerprints matched!")
else:
    print("Fingerprints do not match.")
