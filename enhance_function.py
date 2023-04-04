import cv2
import numpy as np

# def enhance_fingerprint(img):
#     # Convert the image to grayscale
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Apply adaptive thresholding to enhance the ridges
#     img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)

#     # Apply morphological operations to remove noise and fill gaps in the ridges
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
#     img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel)

#     # Apply a median filter to smooth the image and reduce noise
#     img_enhanced = cv2.medianBlur(img_close, 3)

#     return img_enhanced

def enhance_fingerprint(img):
    if len(img.shape) > 2 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    # Your code to enhance the fingerprint image goes here...

    return img_gray

# enhancing the fingerprint image using CLAHE
def enhance_fingerprint_1(img):
    # check the number of channels in the input image
    if len(img.shape) == 2:
        # if the image has only one channel, convert it to a three-channel BGR image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        # if the image has an alpha channel, remove it
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # apply CLAHE to enhance the image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_enhanced = clahe.apply(img_gray)

    return img_enhanced


# enhancing fingerprint images using Histogram Equalization
# this is the best
def enhance_fingerprint_2(img):
    # convert to grayscale if necessary
    if len(img.shape) == 3 and img.shape[2] > 1:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    # apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)
    
    # apply Gaussian blur
    img_blur = cv2.GaussianBlur(img_clahe, (5, 5), 0)
    
    # return the enhanced image
    return img_blur



# uses histogram equalization
def enhance_fingerprint_3(img):
    # check if the input image is already in grayscale
    if len(img.shape) == 2:
        img_gray = img
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply histogram equalization
    img_eq = cv2.equalizeHist(img_gray)

    # apply Gaussian blur
    img_blur = cv2.GaussianBlur(img_eq, (5,5), 0)

    return img_blur

# using the SURF (Speeded-Up Robust Features) algorithm
# def enhance_fingerprint_4(img):

#     # img = cv2.resize(img, (400, 400)) # Resize the image to a smaller size

#     img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
#     orb = cv2.ORB_create()
#     keypoints, descriptors = orb.detectAndCompute(img, None)
#     return descriptors


#  combines multiple techniques
def enhance_fingerprint_m(img):
    if len(img.shape) > 2 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)

    # Apply Median Blur to remove salt-and-pepper noise
    img_median = cv2.medianBlur(img_clahe, 3)

    # Apply Laplacian Filter to enhance edges
    img_lap = cv2.Laplacian(img_median, cv2.CV_64F, ksize=5)
    img_lap = np.uint8(np.absolute(img_lap))

    # Apply Gaussian Blur to smooth the image
    img_blur = cv2.GaussianBlur(img_lap, (5,5), 0)

    return img_blur



def enhance_fingerprint_a(img):
    # convert to grayscale if necessary
    if len(img.shape) == 3 and img.shape[2] > 1:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    # find the center of the fingerprint
    contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        # if no contours are found, use the center of the image
        cx = img_gray.shape[1] // 2
        cy = img_gray.shape[0] // 2
    
    # calculate the size of the square bounding box around the center
    box_size = min(cx, img_gray.shape[1] - cx, cy, img_gray.shape[0] - cy)
    
    # crop the image to the square bounding box
    img_cropped = img_gray[cy - box_size:cy + box_size, cx - box_size:cx + box_size]
    
    # resize the cropped image to a fixed size
    size = (256, 256)  # choose a fixed size
    img_resized = cv2.resize(img_cropped, size)
    
    # apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_resized)
    
    # apply Gaussian blur
    img_blur = cv2.GaussianBlur(img_clahe, (5, 5), 0)
    
    # return the enhanced image
    return img_blur

def enhance_fingerprint_n(img):
    # convert to grayscale if necessary
    if len(img.shape) == 3 and img.shape[2] > 1:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    # find the center of the fingerprint
    contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        # if no contours are found, use the center of the image
        cx = img_gray.shape[1] // 2
        cy = img_gray.shape[0] // 2
    
    # calculate the size of the square bounding box around the center
    box_size = min(cx, img_gray.shape[1] - cx, cy, img_gray.shape[0] - cy)
    
    # crop the image to the square bounding box
    img_cropped = img_gray[cy - box_size:cy + box_size, cx - box_size:cx + box_size]
    
    # resize the cropped image to a fixed size
    size = (256, 256)  # choose a fixed size
    img_resized = cv2.resize(img_cropped, size)
    
    # apply adaptive histogram equalization (AHE)
    img_ahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img_resized)
    
    # apply local contrast normalization (LCN)
    img_lcn = cv2.normalize(img_ahe, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img_lcn = cv2.GaussianBlur(img_lcn, (5, 5), 0)
    
    # apply ridge orientation estimation and correction
    block_size = 16
    gradient_sigma = 3
    ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create(block_size, gradient_sigma)
    orientations = ridge_filter.getRidgeFilteredImage(img_lcn)
    corrected_orientations = ridge_filter.getRidgeFilteredImage(img_lcn)
    radians = corrected_orientations * (np.pi / 180)
    cos_angles = np.cos(2 * radians)
    sin_angles = np.sin(2 * radians)
    img_oriented = np.stack([img_lcn * cos_angles, img_lcn * sin_angles], axis=2)
    
    # return the enhanced image
    return img_oriented
