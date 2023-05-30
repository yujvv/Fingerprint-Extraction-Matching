import cv2
import numpy as np
# import hashlib
# import enhance_function
import adjust7
import os
import crypto
from Crypto.Random import get_random_bytes
import permute_vector
import sys
import matplotlib.pyplot as plt

def plot_dual_line_graph(x_values, y1_values, y2_values, x_label, y1_label, y2_label):
    fig, ax1 = plt.subplots()
    ax1.plot(x_values, y1_values, 'b-')
    ax1.set_ylabel(y1_label+ ' (%)', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(x_values, y2_values, 'r-')
    ax2.set_ylabel(y2_label, color='r')
    ax2.tick_params('y', colors='r')

    # plt.title(title)
    plt.xticks([])
    # plt.legend(loc='upper left')
    plt.show()


def plot_custom_scatter(x_values, y_values):
    fig, ax = plt.subplots()

    # Create a scatter plot with custom marker sizes and colors
    marker_sizes = np.random.randint(10, 100, len(x_values))
    marker_colors = np.random.rand(len(x_values))

    ax.scatter(x_values, y_values, s=marker_sizes, c=marker_colors, cmap='cool', alpha=0.7)

    ax.set_xlabel('Feature size')
    ax.set_ylabel('Compression ratio (%)')

    # Create a colorbar to show the mapping between marker colors and y_values
    sm = plt.cm.ScalarMappable(cmap='cool', norm=plt.Normalize(vmin=min(y_values), vmax=max(y_values)))
    sm.set_array([])  # empty array required for the colorbar to show
    cbar = plt.colorbar(sm)

    plt.show()

def calculate_compression_ratio(original_size, compressed_size):
    compression_ratio = (compressed_size / original_size) * 100
    return compression_ratio

def fingerprint_match_test(ratio, ress:list, orgi:list):
    directory = "enhanced/"
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    for i in range(len(image_files)):
        # Load the fingerprint images
        img1 = cv2.imread(image_files[i], cv2.IMREAD_GRAYSCALE)

        (kp1, des1) = adjust7.extracting(adjust7.enhance_fingerprint(img1))

        (kp1_2, des1_2) = adjust7.filter_2(kp1, des1, img_shape = img1.shape)
        
        compressed_bytes = crypto.compress(des1_2)

        # print("Compressed Bytes_len--------------", len(compressed_bytes))


        # key = get_random_bytes(16)  # 128-bit key
        # # Encrypt the plaintext
        # ciphertext = crypto.encrypt(key, compressed_bytes)
        # # print("Ciphertext: ", ciphertext)

        des_size = sys.getsizeof(des1) + sys.getsizeof(kp1)
        des_size2 = sys.getsizeof(des1_2)
        comp_size = sys.getsizeof(compressed_bytes)
        
        ratio = calculate_compression_ratio(des_size, comp_size)
        ress.append(ratio)
        orgi.append(des_size)
        # print(f"compressed_rate: {ratio:.2f}%")
    
        # print("Ciphertext_size------------------- ", sys.getsizeof(ciphertext))
        # compressed_bytes_size = sys.getsizeof(compressed_bytes)
        # print("compressed_bytes_size------------------- ", compressed_bytes_size)

        # des_size = sys.getsizeof(des1) + sys.getsizeof(kp1)
        # print("des_size------------------- ", des_size)


        # file_size = os.path.getsize(image_files[i])
        # # if os.path.exists(file1):
        # print("file_size------------------- ", file_size)

        # ratio_fg = calculate_compression_ratio(file_size, compressed_bytes_size)
        # print(f"compressed_fg_rate: {ratio_fg:.2f}%")

        # ratio = calculate_compression_ratio(des_size, compressed_bytes_size)
    # return len(image_files)

ratio = 0
ress = []
orgi = []
fingerprint_match_test(ratio, ress, orgi)


x_values = list(range(1, 63))
y1_values = ress
y2_values = orgi
x_label = 'Sample number'
y_label = 'Compression ratio'
y2_label = 'Feature size'

# print("mean rate", np.mean(y1_values))
# print("orgrin size", np.mean(y2_values))
# print("max rate", max(y1_values))

plot_custom_scatter(y2_values, y1_values)

# plot_dual_line_graph(x_values, y1_values, y2_values, x_label, y_label, y2_label)

# print("--------",lent)
# print(f"compressed_rate: {lent:.2f}%")







