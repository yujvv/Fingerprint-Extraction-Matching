import fingerprint_feature_extractor
import cv2
import numpy as np

# Read the fingerprint images
img1 = cv2.imread('enhanced/1.jpg', 0)
img2 = cv2.imread('enhanced/2.jpg', 0)

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

# Create a list to hold the hash values
hash_values = []

# Divide each fingerprint into small blocks and hash each block separately
block_size = 100
for x in range(0, img1.shape[0], block_size):
    for y in range(0, img1.shape[1], block_size):
        # Get the block for each fingerprint
        block1 = img1[x:x+block_size, y:y+block_size]
        block2 = img2[x:x+block_size, y:y+block_size]

        # Calculate the SIFT descriptors for the keypoints in each block
        _, des_block1 = sift.compute(block1, kp1)
        _, des_block2 = sift.compute(block2, kp2)

        # Create a hash object for each block and hash the SIFT descriptors
        block_hash1 = hash(str(des_block1))
        block_hash2 = hash(str(des_block2))

        # Append the hash values to the hash list
        hash_values.append(block_hash1)
        hash_values.append(block_hash2)

# Combine the hash values for both fingerprints into a single list
hash_values_combined = hash_values[:len(hash_values)//2] + hash_values[len(hash_values)//2:]

# Create a set of the hash values to eliminate duplicates
unique_hash_values = set(hash_values_combined)

# Calculate the number of unique hash values
num_unique_hash_values = len(unique_hash_values)

# Print the number of unique hash values
print("Number of unique hash values:", num_unique_hash_values)

# Determine if the fingerprints match based on the number of unique hash values
if num_unique_hash_values > 0:
    print("Fingerprints matched!")
else:
    print("Fingerprints do not match.")
