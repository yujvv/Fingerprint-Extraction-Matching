import cv2
import numpy as np

# Load the fingerprint image
img = cv2.imread("enhanced/1__M_Left_index_finger_CR.BMP", 0)

# Define the ORB detector and descriptor
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
kp, des = orb.detectAndCompute(img, None)

# Draw keypoints on the image
img_kp = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

# Enhance the image by thresholding
thresh_value = 80
img_thresh = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)[1]

# Apply morphological operations to remove noise and fill gaps
kernel = np.ones((5,5), np.uint8)
img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

# Perform histogram equalization to improve contrast
img_eq = cv2.equalizeHist(img_morph)

# Display the enhanced image
cv2.imshow("Enhanced Image", img_eq)
cv2.waitKey(0)
cv2.destroyAllWindows()
