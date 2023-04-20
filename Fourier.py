import numpy as np
from scipy.fftpack import dct, idct

# Generate a random secret key
secret_key = np.random.rand(128)

# Generate a random 128-dimensional vector of integers between 0 and 200
biometric_data = np.random.randint(low=0, high=201, size=128, dtype=np.int16)

# Extract features using the discrete cosine transform (DCT)
features = dct(biometric_data, norm='ortho')

# Add variability by applying the secret key to the features
varied_features = features + secret_key

# Store the cancelable template as the inverse Fourier transform of the varied features
cancelable_template = idct(varied_features, norm='ortho')

# To regenerate the template, apply the DCT to the cancelable template
regenerated_features = dct(cancelable_template, norm='ortho')

# Remove the secret key to obtain the original features
original_features = regenerated_features - secret_key

# Round the original features to the nearest integer and convert back to np.int16
original_features = np.round(original_features).astype(np.int16)

# Invert the DCT to obtain the original biometric data
original_biometric_data = idct(original_features, norm='ortho')

# Check if the original and regenerated biometric data are the same
print(np.array_equal(biometric_data, original_biometric_data))
