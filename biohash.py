import numpy as np
import hashlib

def biohashing(points, seed):
    # Convert the 2D array of minutiae points into a 1D array
    flattened_points = points.flatten()
    # Generate a random seed value for the hash function
    np.random.seed(seed)
    # Generate a random projection matrix with the same dimensions as the flattened array
    projection_matrix = np.random.randn(flattened_points.shape[0], 256)
    # Project the minutiae points onto the projection matrix
    projected_points = np.dot(flattened_points, projection_matrix)
    # Apply the sign function to the projected points to obtain a binary string
    binary_string = np.sign(projected_points)
    # Convert the binary string to a hexadecimal string
    hex_string = binary_string.tobytes().hex()
    # Generate a hash value from the hexadecimal string
    hash_value = hashlib.sha256(bytes.fromhex(hex_string)).hexdigest()
    # Return the hash value as the cancelable template
    return hash_value

# Example usage
minutiae_points = np.array([[10, 20], [30, 40], [50, 60], [70, 80]])
seed = 123456789
cancelable_template = biohashing(minutiae_points, seed)
print(cancelable_template)

