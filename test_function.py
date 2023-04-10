import cv2
import numpy as np
# import hashlib
# import enhance_function
import adjust7
import os

def fingerprint_match_test(similarity_score_boundary):
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

            # img1 = cv2.imread(file1, cv2.IMREAD_GRAYSCALE)
            # img2 = cv2.imread(file2, cv2.IMREAD_GRAYSCALE)

            (kp1, des1) = adjust7.extracting(adjust7.enhance_fingerprint(img1))
            (kp2, des2) = adjust7.extracting(adjust7.enhance_fingerprint(img2))
            (kp1_2, des1_2) = adjust7.filter_2(kp1, des1, img_shape = img1.shape)
            (kp2_2, des2_2) = adjust7.filter_2(kp2, des2, img_shape = img1.shape)
            # (kp1_, des1_) = filter(kp1, des1)
            # (kp2_, des2_) = filter(kp2, des2)
            similarity_score = adjust7.matching(des1_2, des2_2)

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

    # print(f"Number of comparisons: {num_comparisons}")
    # print(f"Number of successful matches: {num_successes}")
    # print(f"False acceptance rate (FAR): {FAR:.4f}")
    # print(f"False rejection rate (FRR): {FRR:.4f}")
    # print(f"Error rate (ERR): {ERR:.4f}")

    # with open("output.txt", "w") as file:
    with open("output.txt", "a") as file:
        file.write(f"similarity_score_boundary: {similarity_score_boundary}\n")
        file.write(f"\n")
        file.write(f"Number of successful matches: {num_successes}\n")
        file.write(f"False acceptance rate (FAR): {FAR:.4f}\n")
        file.write(f"False rejection rate (FRR): {FRR:.4f}\n")
        file.write(f"Error rate (ERR): {ERR:.4f}\n")
        file.write(f"-------------------\n")
