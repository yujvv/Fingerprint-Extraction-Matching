import random
import hashlib
import numpy as np


def generate_password(length=16):
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+"
    password = "".join(random.choice(characters) for i in range(length))
    return password

def hash_password(password):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    seed = int(hashed[:8], 16)
    return str(seed)




def scramble_array(array, password):
    hashed_password = hashlib.sha256(password.encode()).digest()
    seed = int.from_bytes(hashed_password, byteorder='big') % (2**32 - 1)
    np.random.seed(seed)
    np.random.shuffle(array)
    return array

# Example usage
password = generate_password()
array = np.random.randint(0, 256, size=(10, 128))
scrambled_array = scramble_array(array, password)
print('----------------',array)
print(scrambled_array)
