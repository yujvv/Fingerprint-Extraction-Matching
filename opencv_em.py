import cv2
import numpy as np

img1 = cv2.imread('enhanced/2.jpg', 0)
img2 = cv2.imread('enhanced/2.jpg', 0)

# Perform template matching
res = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)

# Set threshold and find matched points
threshold = 0.8
loc = np.where(res >= threshold)
if len(loc[0]) > 0:
    print("Fingerprints matched!")
else:
    print("Fingerprints do not match.")

# In this code, we first read in the two fingerprint images img1 and img2. Then, we use the cv2.matchTemplate function to perform template matching. This function returns a 2D array of floating-point values, where each value represents the correlation between the template image (in this case, img2) and a corresponding region of the search image (in this case, img1).

# We then set a threshold value (0.8 in this case) and use the np.where function to find the locations where the correlation values are greater than or equal to this threshold. If any such locations are found, we print "Fingerprints matched!", otherwise we print "Fingerprints do not match."