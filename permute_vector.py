import numpy as np
import hashlib
import random


def scramble_vector(vector, password = "LtwHqkXsH5HCnViF"):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    seed = int(hashed_password[:8], 16)
    np.random.seed(seed)
    scrambled_vector = np.random.permutation(vector)
    return scrambled_vector

# Generate a random password
# password = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=16))


# Generate a 128-bit vector of random integers between 0 and 255
# vector = np.random.randint(256, size=128)

# print("Password:", password)
# print("Vector:", vector)

# print('---------------------', scramble_vector(vector, password))