# This code uses the MinutiaNet feature extractor to extract minutiae features from the fingerprints, and then uses a simple distance-based matching algorithm to match the features.
# install the MinutiaNet package (pip install minutianet) and download the pre-trained weights (weights='pretrained').
import cv2
import numpy as np
import tensorflow as tf
from minutianet import MinutiaNet

# Load the MinutiaNet model
model = MinutiaNet(weights='pretrained')

# Load the images
img1 = cv2.imread('enhanced/2.jpg', 0)
img2 = cv2.imread('enhanced/1.jpg', 0)

# Preprocess the images
img1 = cv2.equalizeHist(img1)
img2 = cv2.equalizeHist(img2)

# Resize the images to 256x256
img1 = cv2.resize(img1, (256, 256))
img2 = cv2.resize(img2, (256, 256))

# Extract minutiae features using MinutiaNet
features1 = model.predict(np.expand_dims(img1, axis=0))
features2 = model.predict(np.expand_dims(img2, axis=0))

# Matching minutiae features
matched_features = []
for i in range(features1.shape[1]):
    for j in range(features2.shape[1]):
        dist = np.linalg.norm(features1[0,i,:] - features2[0,j,:])
        if dist < 0.1:  # distance threshold
            matched_features.append((int(features1[0,i,0]*img1.shape[1]), int(features1[0,i,1]*img1.shape[0]), int(features2[0,j,0]*img2.shape[1]), int(features2[0,j,1]*img2.shape[0])))

if len(matched_features) > 0:
    print("Fingerprints matched!")
else:
    print("Fingerprints do not match.")
