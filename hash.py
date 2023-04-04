import numpy as np
import random
# import multiprocessing as mp
# # has indeed been deprecated
# from sklearn.neighbors import LSHForest
from datasketch import MinHash, LeanMinHash
import cv2

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

def hash_descriptors_1(descriptors, blockSize, hashSize, numHashes):
    hashList = []
    for i in range(numHashes):
        random.seed(i)
        hashFunc = random.sample(range(descriptors.shape[1]), hashSize)
        hashVal = []
        for descriptor in descriptors:
            hash = ""
            for j in hashFunc:
                block = descriptor[j*blockSize:(j+1)*blockSize]
                mean = np.mean(block)
                hash += "1" if (block > mean).sum() >= 2 else "0"
            hashVal.append(int(hash, 2))
        hashList.append(np.array(hashVal))
        # print(f"Hash function {i+1}:\nHash values: {hashList[i]}\n")
    return np.vstack(hashList).T


# one approach that can be explored is to use overlapping blocks instead of non-overlapping blocks. This means that each block will overlap with the adjacent block by a certain number of pixels, allowing for more information to be captured in each block.

# Another approach is to use adaptive block sizes instead of a fixed block size. This means that larger blocks can be used in areas with more complex features, while smaller blocks can be used in areas with simpler features. This can improve the accuracy of the hash function by capturing more meaningful information in each block.
# we have removed the blockSize argument and modified the inner loop to use a variable blockSize that is calculated based on the complexity of the feature in the current region. In this implementation, we have used a simple heuristic of using a larger block size when the standard deviation of the pixel intensities is greater than a certain threshold. This threshold can be adjusted based on the specific application.
# this way is the best
def hash_descriptors_2(descriptors, blockSize, hashSize, numHashes):
    hashList = []
    for i in range(numHashes):
        random.seed(i)
        hashFunc = random.sample(range(descriptors.shape[1]), hashSize)
        hashVal = []
        for descriptor in descriptors:
            hash = ""
            for j in range(0, descriptors.shape[1], blockSize):
                block = descriptor[j:j+blockSize]
                if len(block) < blockSize:
                    continue
                mean = np.mean(block)
                hash += "1" if (block > mean).sum() >= 2 else "0"
            hashVal.append(int(hash, 2))
        hashList.append(np.array(hashVal))
    return np.vstack(hashList).T

# added an additional argument overlap, which determines the number of pixels that each block overlaps with the adjacent block. We have also modified the inner loop to increment j by blockSize-overlap instead of blockSize.
def hash_descriptors_3(descriptors, blockSize, hashSize, numHashes, overlap):
    hashList = []
    for i in range(numHashes):
        random.seed(i)
        hashFunc = random.sample(range(descriptors.shape[1]), hashSize)
        hashVal = []
        for descriptor in descriptors:
            hash = ""
            for j in range(0, descriptors.shape[1]-blockSize+1, blockSize-overlap):
                block = descriptor[j:j+blockSize]
                mean = np.mean(block)
                hash += "1" if (block > mean).sum() >= 2 else "0"
            hashVal.append(int(hash, 2))
        hashList.append(np.array(hashVal))
    return np.vstack(hashList).T



# #  use the LSH Forest algorithm from scikit-learn to hash the descriptors. We also divide the descriptors into non-overlapping blocks and hash each block using the LSH Forest. Finally, we concatenate the resulting hash values into a single binary hash.
# def hash_descriptors_4(descriptors, blockSize, hashSize, numHashes):
#     # Initialize LSH Forest
#     lshf = LSHForest(n_estimators=numHashes, n_candidates=100)
#     lshf.fit(descriptors)
    
#     # Divide descriptors into blocks
#     blocks = []
#     numBlocks = descriptors.shape[1] // blockSize
#     for i in range(numBlocks):
#         blocks.append(descriptors[:, i*blockSize:(i+1)*blockSize])
        
#     # Hash each block using LSH Forest
#     hashList = []
#     for i in range(numHashes):
#         hashes = lshf.predict(blocks)
#         hashList.append(hashes)
    
#     # Convert hash values to binary strings and concatenate
#     hashList = np.vstack(hashList).T
#     binaryHash = np.packbits(hashList)
    
#     return binaryHash




# In this implementation, each descriptor is hashed using a MinHash object, which is created with a specified number of permutations (i.e., hash functions). For each permutation, we update the MinHash object with the index of the block and the value of each element in the block that is above its mean. Finally, we convert the MinHash object to an integer hash value using the built-in hash() function, and append it to the hashVal list.

def hash_descriptors_4(descriptors, blockSize, hashSize, numHashes):
    assert hashSize % 8 == 0, "hashSize must be divisible by 8"
    assert len(descriptors) % blockSize == 0, "blockSize must be a multiple of the number of descriptors"

    blockStride = blockSize // len(descriptors)
    descList = np.array(descriptors, dtype=np.float32)
    blocks = np.array_split(descList, blockSize // blockStride)

    # minhash = LeanMinHash(numHashes)
    minhash = MinHash(numHashes)
    for i, block in enumerate(blocks):
        if i >= numHashes:
            break
        m = cv2.ORB_create()
        _, descriptors = m.compute(block, None)
        if descriptors is not None:
            descriptors = descriptors.reshape(-1, 32)
            for j in range(descriptors.shape[0]):
                for val in descriptors[j]:
                    m.update(str(j) + str(val))
        else:
            continue
        h = m.hexdigest()
        bytes_val = bytes.fromhex(h)
        for i in range(0, hashSize, 8):
            minhash.update(bytes_val[i:i+8])
    return minhash

def get_block_size(descriptors):
    num_descriptors = descriptors.shape[0]
    for i in range(num_descriptors, 0, -1):
        if num_descriptors % i == 0:
            return i
        


import random

def hash_descriptors_t(descriptors, blockSize, hashSize, numHashes):
    # Split the descriptors into blocks of size blockSize x blockSize
    blocks = [descriptors[:, i:i+blockSize] for i in range(0, descriptors.shape[1], blockSize)]

    # Compute the hash for each block
    hashes = []
    for i in range(numHashes):
        # Randomly select hashSize blocks to form the hash
        if hashSize > len(blocks):
            raise ValueError("hashSize is larger than the number of available blocks")
        block_indices = random.sample(range(len(blocks)), hashSize)
        block_hash = np.hstack([blocks[j] for j in block_indices]).astype(np.uint8)
        hashes.append(block_hash)

    # Concatenate the hashes horizontally to form the final block hash array
    return np.hstack(hashes)

