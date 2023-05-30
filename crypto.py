import zlib
import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import base64

# # Define a sample 2D array
# input_array = np.array([[1, 2, 3, 4],
#                        [5, 6, 7, 8],
#                        [9, 10, 11, 12]])

def compress(input_array):
    # Convert the 2D array to bytes
    input_bytes = input_array.tobytes()
    # Compress the bytes using Deflate
    compressed_bytes = zlib.compress(input_bytes, zlib.DEFLATED)
    # Print the compressed bytes
    # print("Compressed Bytes: ", compressed_bytes)
    return compressed_bytes

def decompress(compressed_bytes):
    # Decompress the compressed bytes using Deflate
    decompressed_bytes = zlib.decompress(compressed_bytes)
    # Convert the decompressed bytes back to a 2D array
    output_array = np.frombuffer(decompressed_bytes, dtype=input_array.dtype)
    output_array = output_array.reshape(input_array.shape)
    # Print the decompressed 2D array
    print("Decompressed Array: \n", output_array)
    return output_array




# Define the AES encryption function
def encrypt(key, plaintext):
    cipher = AES.new(key, AES.MODE_CBC)
    ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))
    return base64.b64encode(cipher.iv + ciphertext).decode()

# Define the AES decryption function
def decrypt(key, ciphertext):
    ciphertext = base64.b64decode(ciphertext.encode())
    iv = ciphertext[:AES.block_size]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = unpad(cipher.decrypt(ciphertext[AES.block_size:]), AES.block_size)
    return plaintext

import numpy as np

def compress_2d_array(arr, k):
    """Compresses a 2D array using Singular Value Decomposition (SVD).
    
    Args:
        arr (numpy.ndarray): A 2D array.
        k (int): The number of singular values to keep for the compression.
        
    Returns:
        numpy.ndarray: The compressed 2D array.
    """
    # Apply SVD to the array
    U, S, V = np.linalg.svd(arr)
    
    # Truncate the singular values and matrices to keep only the top k values
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    V_k = V[:k, :]
    
    
    # Reconstruct the compressed array
    arr_compressed = U_k @ S_k @ V_k
    
    return arr_compressed

def array_to_string(arr):
    """Converts a 2D array to a string representation.
    
    Args:
        arr (numpy.ndarray): A 2D array.
        
    Returns:
        str: The string representation of the array.
    """
    # Flatten the array and convert it to a list of strings
    arr_flat = arr.flatten().tolist()
    arr_str = [str(x) for x in arr_flat]
    
    # Get the dimensions of the array
    rows, cols = arr.shape
    
    # Build the string representation of the array
    arr_str = ','.join(arr_str)
    arr_str = f'{rows},{cols};{arr_str}'
    
    return arr_str



# # Define the key and plaintext
# key = get_random_bytes(16)  # 128-bit key
# plaintext = compress(input_array)   # Input plaintext string

# # Encrypt the plaintext
# ciphertext = encrypt(key, plaintext)
# print("Ciphertext: ", ciphertext)

# # Decrypt the ciphertext
# decrypted_text = decrypt(key, ciphertext)
# print("Decrypted Text: ", decrypted_text)



# # Create a random 2D array
# arr = np.random.rand(10, 10)

# # Compress the array with k=5
# arr_compressed = compress_2d_array(input_array, k=1)

# # Print the original and compressed arrays
# print("Original array:\n", input_array)
# print("Compressed array:\n", arr_compressed)
