import cv2
import numpy as np
import hashlib
import enhance_function
import os

# similarity_score_boundary = 500
# matching_criteria = 0.75
# block_size_std = 64

def fingerprint_match_test(similarity_score_boundary, matching_criteria, block_size_std):
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
            img1_ = cv2.imread(image_files[i], cv2.IMREAD_GRAYSCALE)
            img2_ = cv2.imread(image_files[j], cv2.IMREAD_GRAYSCALE)

            # a and n are both ok
            img1 = enhance_function.enhance_fingerprint_a(img1_)
            img2 = enhance_function.enhance_fingerprint_a(img2_)

            # Create the SIFT detector object
            sift = cv2.SIFT_create()

            # Detect keypoints and compute descriptors for both images
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            # Divide descriptors into blocks
            block_size = block_size_std  # Number of descriptors per block
            des1_blocks = np.array_split(des1, len(des1) // block_size)
            des2_blocks = np.array_split(des2, len(des2) // block_size)

            # Apply hashing to each block and concatenate hashed values
            hash1_blocks = [hashlib.sha256(block.tobytes()).hexdigest() for block in des1_blocks]
            hash2_blocks = [hashlib.sha256(block.tobytes()).hexdigest() for block in des2_blocks]
            hash1 = ''.join(hash1_blocks)
            hash2 = ''.join(hash2_blocks)

            # print('------hash------',hash1)

            # Divide concatenated hash values into blocks
            hash_block_size = block_size  # Number of characters per block
            hash1_blocks = [hash1[i:i+hash_block_size] for i in range(0, len(hash1), hash_block_size)]
            hash2_blocks = [hash2[i:i+hash_block_size] for i in range(0, len(hash2), hash_block_size)]

            # Create a FLANN object
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            # Build the FLANN index from the descriptors of the second image
            des2_float = np.float32(des2)
            flann.add([des2_float])

            # Compare blocks using the FLANN index and ratio test
            similarity_score = 0
            for i in range(len(des1_blocks)):
                des1_block = des1_blocks[i]
                if i < len(des2_blocks):
                    des2_block = des2_blocks[i]
                    matches = flann.knnMatch(des1_block, k=2)
                    good_matches = []
                    for m, n in matches:
                        if m.distance < matching_criteria*n.distance:
                            good_matches.append(m)
                    similarity_score += len(good_matches)

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
        file.write(f"block_size: {block_size_std}\n")
        file.write(f"\n")
        file.write(f"Number of comparisons: {num_comparisons}\n")
        file.write(f"Number of successful matches: {num_successes}\n")
        file.write(f"False acceptance rate (FAR): {FAR:.4f}\n")
        file.write(f"False rejection rate (FRR): {FRR:.4f}\n")
        file.write(f"Error rate (ERR): {ERR:.4f}\n")
        file.write(f"-------------------\n")
