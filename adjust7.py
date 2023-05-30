import cv2
import numpy as np
import crypto
from Crypto.Random import get_random_bytes
import permute_vector

file1 = 'enhanced/103_1.tif'
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

    return [similarity_score,good_matches]



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








# img1 = cv2.imread(file1, cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread(file2, cv2.IMREAD_GRAYSCALE)

# (kp1, des1) = extracting(enhance_fingerprint(img1))
# (kp2, des2) = extracting(enhance_fingerprint(img2))

# # img_enhanced = enhance_fingerprint(img1)
# # # Display the enhanced image
# # cv2.imshow('Enhanced Fingerprint', img1)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# (kp1_2, des1_2) = filter_2(kp1, des1, img_shape = img1.shape)
# (kp2_2, des2_2) = filter_2(kp2, des2, img_shape = img1.shape)

# # Display the image with keypoints
# # img_with_keypoints = cv2.drawKeypoints(img1, kp1_2, None)
# # cv2.imshow('Fingerprint with keypoints', img_with_keypoints)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # (kp1_, des1_) = filter(kp1, des1)
# # (kp2_, des2_) = filter(kp2, des2)

# # print("-------des1_2 before----------------------------", des1_2)

# for i, row in enumerate(des1_2):
#     des1_2[i] = permute_vector.scramble_vector(row, "LtwHqkXsH5HCnViF")

# for i, row in enumerate(des2_2):
#     des2_2[i] = permute_vector.scramble_vector(row, "LtwHqkXsH5HCnViF")

# # print("-------des1_2 after----------------------------", des1_2)

# similarity_score = matching(des1_2, des2_2)

# # print("--------len_des1--------", len(des1))
# # print("--------len_des1_--------", len(des1_))



# print("--------len_des1_2--------", len(des1_2))
# print("----------------", similarity_score[0])

# # # Draw the matches on a new image
# # img_matches = cv2.drawMatches(img1, kp1, img2, kp2, similarity_score[1], None)

# # # Display the image
# # cv2.imshow('Matches', img_matches)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()



# print("des_len--------------", len(crypto.array_to_string(des1_2)))


# des_svd = crypto.compress_2d_array(des1_2, k=5)
# compressed_bytes_svd = crypto.compress(des_svd)
# print("compressed_bytes_svd_len--------------", len(compressed_bytes_svd))

# compressed_bytes = crypto.compress(des1_2)
# print("Compressed Bytes_len--------------", len(compressed_bytes))


# key = get_random_bytes(16)  # 128-bit key
# # Encrypt the plaintext
# ciphertext = crypto.encrypt(key, compressed_bytes)
# # print("Ciphertext: ", ciphertext)

# import sys
# print("Ciphertext_size------------------- ", sys.getsizeof(ciphertext))
# compressed_bytes_size = sys.getsizeof(compressed_bytes)
# print("compressed_bytes_size------------------- ", compressed_bytes_size)

# des_size = sys.getsizeof(des1) + sys.getsizeof(kp1)
# print("des_size------------------- ", des_size)

# import os
# file_size = os.path.getsize(file1)
# # if os.path.exists(file1):
# print("file_size------------------- ", file_size)

# def calculate_compression_ratio(original_size, compressed_size):
#     compression_ratio = (compressed_size / original_size) * 100
#     return compression_ratio

# ratio_fg = calculate_compression_ratio(file_size, compressed_bytes_size)
# print(f"compressed_fg_rate: {ratio_fg:.2f}%")

# ratio = calculate_compression_ratio(des_size, compressed_bytes_size)
# print(f"compressed_rate: {ratio:.2f}%")