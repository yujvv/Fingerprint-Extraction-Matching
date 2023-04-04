import os
import cv2
import numpy as np
import random

# Initialize feature detector and descriptor
detector = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

directory = "enhanced/"
image_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# num_false_accepts is the number of pairs that were falsely accepted as a match, num_false_rejects is the number of pairs that were falsely rejected as a non-match, and num_comparison_pairs is the total number of comparison pairs
num_true_matches = 0
num_false_matches = 0
num_false_nonmatches = 0

def fingerprint_match(fp1, fp2):
    """
    Determines if two fingerprints match based on the first few digits of their filenames.
    """
    fp1_prefix = os.path.basename(fp1).split("_")[:4]
    fp2_prefix = os.path.basename(fp2).split("_")[:4]
    return fp1_prefix == fp2_prefix


for i in range(len(image_files)):
    for j in range(i+1, len(image_files)):
        img1 = cv2.imread(image_files[i], 0)
        img2 = cv2.imread(image_files[j], 0)

        sameFinger = fingerprint_match(image_files[i], image_files[j])

        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)

        # Hash the descriptors, size of blocks, numHashes random hash functions
        blockSize = 4
        hashSize = 12
        # Each hash function selects hashSize blocks from the descriptors
        numHashes = 3

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

        # Define the threshold value for matching
        match_threshold = 10

        # Apply the ratio test to filter out false matches
        matches = sorted(matches, key=lambda x: x.distance)
        # The ratio test is applied to filter out false matches by sorting the matches by their distance and keeping only the matches that have a distance ratio lower than 0.75.
        ratio = 0.75
        num_good_matches = int(len(matches) * ratio)
        matches = matches[:num_good_matches]


        # Check if the fingerprints match based on the number of good matches
        if num_good_matches >= match_threshold:
            if i == j or sameFinger:
                num_true_matches += 1
            else:
                num_false_matches += 1
        else:
            if i == j or sameFinger:
                num_false_nonmatches += 1


num_comparison_pairs = len(image_files) * (len(image_files) - 1) // 2
FAR = num_false_matches / num_comparison_pairs
FRR = num_false_nonmatches / num_comparison_pairs
EER = (FAR + FRR) / 2

print("FAR:",FAR)
print("FRR:",FRR)
print("EER:",EER)