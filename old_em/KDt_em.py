import fingerprint_feature_extractor
import cv2
import numpy as np
from sklearn.neighbors import KDTree

img1 = cv2.imread('enhanced/1.jpg', 0)
img2 = cv2.imread('enhanced/2.jpg', 0)

# Extracting minutiae features
FeaturesTerminations1, FeaturesBifurcations1 = fingerprint_feature_extractor.extract_minutiae_features(img1, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False)
FeaturesTerminations2, FeaturesBifurcations2 = fingerprint_feature_extractor.extract_minutiae_features(img2, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False)

# Convert minutiae features to arrays for KDTree
X1 = np.array([(t.locX, t.locY) for t in FeaturesTerminations1] + [(b.locX, b.locY) for b in FeaturesBifurcations1])
X2 = np.array([(t.locX, t.locY) for t in FeaturesTerminations2] + [(b.locX, b.locY) for b in FeaturesBifurcations2])

# Build KDTree for image 2
tree = KDTree(X2)

# Match minutiae features
matched_features = []
for i, (x1, y1) in enumerate(X1):
    ind = tree.query_radius([(x1, y1)], r=5)[0]
    for j in ind:
        x2, y2 = X2[j]
        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        if dist < 5:
            matched_features.append((x1, y1, x2, y2))
            break

if len(matched_features) > 0:
    print("Fingerprints matched!")
else:
    print("Fingerprints do not match.")
