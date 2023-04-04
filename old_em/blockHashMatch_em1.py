# we loop through multiple hash algorithms and use each one to generate a hash value for each feature descriptor. We then concatenate all the hash values to produce the encrypted feature descriptor. We do this separately for both images before matching them using the same BFMatcher and ratio test as before.
import cv2
import numpy as np
import hashlib

# Define the hash algorithms to use
hash_algorithms = [
    hashlib.sha1,
    hashlib.sha224,
    hashlib.sha256,
    hashlib.sha384,
    hashlib.sha512,
]

# Load the images
img1 = cv2.imread('enhanced/1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('enhanced/1.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize the ORB detector and compute the descriptors for each image
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Encrypt the feature descriptors using multiple hash algorithms
enc_des1 = []
for d in des1:
    enc_d = b''
    for hash_alg in hash_algorithms:
        hash_obj = hash_alg()
        hash_obj.update(d.tobytes())
        enc_d += hash_obj.digest()
    enc_des1.append(enc_d)

enc_des2 = []
for d in des2:
    enc_d = b''
    for hash_alg in hash_algorithms:
        hash_obj = hash_alg()
        hash_obj.update(d.tobytes())
        enc_d += hash_obj.digest()
    enc_des2.append(enc_d)

# Initialize the matcher and perform the matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(enc_des1, enc_des2)

# Apply the ratio test to filter out false matches
matches = sorted(matches, key=lambda x: x.distance)
good_matches = []
for m in matches:
    if m.distance < 0.75 * matches[0].distance:
        good_matches.append(m)

# Draw the matches
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('matches.jpg', img3)
