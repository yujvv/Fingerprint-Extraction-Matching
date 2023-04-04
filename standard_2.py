import cv2

# Load the fingerprint images
img1 = cv2.imread('enhanced/103_3.tif', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('enhanced/103_2.tif', cv2.IMREAD_GRAYSCALE)

# Create the SIFT detector object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Create a BFMatcher object
bf = cv2.BFMatcher()

# Match the descriptors using the BFMatcher
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test to filter good matches
good_matches = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_matches.append(m)

print('---------------', len(good_matches))

# # Draw the matches on a new image
# img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)

# # Display the image
# cv2.imshow('Matches', img_matches)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
