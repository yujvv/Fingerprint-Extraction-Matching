import cv2
import numpy as np
import hashlib
import enhance_function
import os

similarity_score_boundary = 140
matching_criteria = 0.75

directory = "enhanced/"
image_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Initialize variables for the false acceptance rate (FAR), false rejection rate (FRR), and error rate (ERR)
num_successes = 0
num_false_acceptances = 0
num_false_rejections = 0
num_comparisons = len(image_files) * (len(image_files) - 1) // 2

def fingerprint_match(fp1, fp2):
    """
    Determines if two fingerprints match based on the first three digits of their filenames.
    """
    fp1_prefix = os.path.basename(fp1)[:3]
    fp2_prefix = os.path.basename(fp2)[:3]
    return fp1_prefix == fp2_prefix

for i in range(len(image_files)):
    for j in range(i+1, len(image_files)):
        # Load the fingerprint images
        img1 = cv2.imread(image_files[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_files[j], cv2.IMREAD_GRAYSCALE)

        # Create the SIFT detector object
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors for both images
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Create a BFMatcher object
        bf = cv2.BFMatcher()

        # Match the descriptors using the BFMatcher
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test to filter good matches
        good_matches = []
        for m,n in matches:
            if m.distance < matching_criteria*n.distance:
                good_matches.append(m)

        similarity_score = len(good_matches)
        # print('Similarity score:', similarity_score)
        samePrint = fingerprint_match(image_files[i], image_files[j])

        if similarity_score>similarity_score_boundary and samePrint:
            num_successes+=1
        elif similarity_score<similarity_score_boundary and not samePrint:
            num_successes+=1
        elif similarity_score>similarity_score_boundary and not samePrint:
            num_false_acceptances+=1
        elif similarity_score<similarity_score_boundary and samePrint:
            num_false_rejections+=1


# Calculate the false acceptance rate (FAR), false rejection rate (FRR), and error rate (ERR)
FAR = num_false_acceptances / num_successes
FRR = num_false_rejections / (num_comparisons - num_successes)
ERR = (num_false_acceptances + num_false_rejections) / num_comparisons

print(f"Number of comparisons: {num_comparisons}")
print(f"Number of successful matches: {num_successes}")
print(f"False acceptance rate (FAR): {FAR:.4f}")
print(f"False rejection rate (FRR): {FRR:.4f}")
print(f"Error rate (ERR): {ERR:.4f}")