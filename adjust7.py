import cv2
import numpy as np
import crypto
from Crypto.Random import get_random_bytes

file1 = 'enhanced/102_3.tif'
file2 = 'enhanced/103_3.tif'

def enhance_fingerprint(img):
    # convert to grayscale if necessary
    if len(img.shape) == 3 and img.shape[2] > 1:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    # apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)
    
    # apply Gaussian blur
    img_blur = cv2.GaussianBlur(img_clahe, (5, 5), 0)
    
    # return the enhanced image
    return img_blur

def extracting(file):
    # Load the fingerprint images
    # img1 = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    # Create the SIFT detector object
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    kp, des = sift.detectAndCompute(file, None)

    return (kp, des)

def matching(des1, des2):
        # Create Flann based matcher object
    flann = cv2.FlannBasedMatcher()

    # Match descriptors of both images using k-NN search (k=2)
    matches = flann.knnMatch(des1, des2, k=2)

    # Filter matches based on Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Compute similarity score
    similarity_score = len(good_matches) / max(len(des1), len(des2))

    return similarity_score



def filter(kp, des):
    # Initialize a list to hold the indices of keypoints to remove
    idx_to_remove = []
    
    # Loop through the keypoints and identify "short ridges" and "islands"
    for i, k in enumerate(kp):
        # If the keypoint is an "island" or a "short ridge", add its index to the list of keypoints to remove
        if k.response < 0.02 or k.size < 4:
            idx_to_remove.append(i)

    # Remove the corresponding descriptors for the identified keypoints
    filtered_des = np.delete(des, idx_to_remove, axis=0)

    # Remove the identified keypoints from the keypoints list
    filtered_kp = [kp[i] for i in range(len(kp)) if i not in idx_to_remove]

    return filtered_kp, filtered_des



def filter_2(kp, des, img_shape):
    # Initialize a list to hold the indices of keypoints to remove
    idx_to_remove = []
    
    # Loop through the keypoints and identify keypoints to remove
    for i, k in enumerate(kp):
        x, y = k.pt
        
        # Check for islands (small areas where the ridges in a fingerprint converge and then diverge again)
        if k.size < 4:
            idx_to_remove.append(i)
            continue
            
        # Check for short ridges (ridge segments that are shorter than the surrounding ridges)
        if k.response < 0.02:
            idx_to_remove.append(i)
            continue
        
        # Check for bifurcations with sharp angles
        # if k.angle < math.pi/4 or k.angle > 3*math.pi/4:
        #     idx_to_remove.append(i)
        #     continue

        # Check for bifurcations with sharp angles
        for i, k in enumerate(kp):
            if k.angle >= 60 and k.class_id == 2:
                idx_to_remove.append(i)
            
        # Check for endings near the edge of the fingerprint
        if x < 20 or y < 20 or x > img_shape[1]-20 or y > img_shape[0]-20:
            idx_to_remove.append(i)
            continue
            
        # Check for spur or delta features
        if k.class_id == 3 or k.class_id == 4:
            idx_to_remove.append(i)
            continue

    # Remove the corresponding descriptors for the identified keypoints
    filtered_des = np.delete(des, idx_to_remove, axis=0)

    # Remove the identified keypoints from the keypoints list
    filtered_kp = [kp[i] for i in range(len(kp)) if i not in idx_to_remove]

    return filtered_kp, filtered_des






img1 = cv2.imread(file1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(file2, cv2.IMREAD_GRAYSCALE)

(kp1, des1) = extracting(enhance_fingerprint(img1))
(kp2, des2) = extracting(enhance_fingerprint(img2))
(kp1_2, des1_2) = filter_2(kp1, des1, img_shape = img1.shape)
(kp2_2, des2_2) = filter_2(kp2, des2, img_shape = img1.shape)

(kp1_, des1_) = filter(kp1, des1)
(kp2_, des2_) = filter(kp2, des2)
similarity_score = matching(des1_, des2_)

print("--------len_des1--------", len(des1))
print("--------len_des1_--------", len(des1_))
print("--------len_des1_2--------", len(des1_2))
print("----------------", similarity_score)

compressed_bytes = crypto.compress(des1_2)
print("Compressed Bytes: ", compressed_bytes)


key = get_random_bytes(16)  # 128-bit key
# Encrypt the plaintext
ciphertext = crypto.encrypt(key, compressed_bytes)
print("Ciphertext: ", ciphertext)
