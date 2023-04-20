import numpy as np
import hashlib

def polar_fourier_transform(x, y, rmax, ntheta, nrho):
    """
    Compute the polar Fourier transform of a 2D signal
    :param x: The x-coordinate of the signal points
    :param y: The y-coordinate of the signal points
    :param rmax: The maximum radius for the polar coordinates
    :param ntheta: The number of angular samples
    :param nrho: The number of radial samples
    :return: The polar Fourier transform of the signal
    """
    # Compute the polar coordinates of the signal points
    rho = np.linspace(0, rmax, nrho)
    theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
    grid_r, grid_theta = np.meshgrid(rho, theta)
    xgrid = grid_r * np.cos(grid_theta)
    ygrid = grid_r * np.sin(grid_theta)
    # Interpolate the signal onto the polar grid
    signal = np.zeros((nrho, ntheta))
    for i in range(ntheta):
        signal[:, i] = np.interp(rho, np.sqrt((x - xgrid[:, i])**2 + (y - ygrid[:, i])**2), np.ones_like(rho))
    # Compute the polar Fourier transform of the signal
    F = np.fft.fft(signal, axis=0)
    return F.flatten()

def pft_kdt_template(description, key, rmax=0.5, ntheta=16, nrho=32):
    """
    Generate a cancellable fingerprint template using the PFT-KDT method
    :param description: The feature descriptor array for the fingerprint
    :param key: The secret key used to parameterize the transform
    :param rmax: The maximum radius for the polar coordinates (default: 0.5)
    :param ntheta: The number of angular samples (default: 16)
    :param nrho: The number of radial samples (default: 32)
    :return: The cancellable fingerprint template
    """
    # Compute the hash of the key
    key_hash = hashlib.sha256(key.encode('utf-8')).digest()
    # Compute the polar Fourier transform of the feature descriptors
    pft = polar_fourier_transform(description[:, 0], description[:, 1], rmax=rmax, ntheta=ntheta, nrho=nrho)
    # Apply the key-dependent random projection to the transformed data
    rng = np.random.default_rng(int.from_bytes(key_hash, byteorder='big'))
    projection = rng.standard_normal((pft.shape[0], pft.shape[0]//2))
    projection /= np.sqrt(np.sum(projection**2, axis=0, keepdims=True))
    projection *= np.sqrt(pft.shape[0])
    template = np.matmul(pft, projection)
    # Return the cancellable fingerprint template
    return template



import secrets

key = secrets.token_hex(16)
print(key)

# Example usage
description = np.random.rand(10, 2)  # Example feature descriptor array

template = pft_kdt_template(description, key)  # Generate the cancellable fingerprint template
