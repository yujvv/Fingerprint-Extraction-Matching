import numpy as np
import cv2
from PIL import Image
import numpy as np
import math

def thinning(img):
    # Zhang-Suen thinning algorithm
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)

    ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False

    while(not done):
        eroded = cv2.erode(img,element)
        dilated = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,dilated)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True

    return skel

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def extract_minutiae_from_skeleton(img_skeleton):
    # Find contours in the skeleton image
    contours, hierarchy = cv2.findContours(img_skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    # Approximate the contour to remove self-intersections
    epsilon = 0.1 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    # Compute the convex hull of the contour
    hull = cv2.convexHull(approx, returnPoints=False)

    # Compute the convexity defects
    defects = cv2.convexityDefects(approx, hull)

    if defects is None:
        return []  # Return an empty list if no defects are found

    # Extract the minutiae from the convexity defects
    minutiae = []
    MIN_DISTANCE = 10
    for i in range(defects.shape[0]):
        start, end, far, _ = defects[i][0]
        start_point = tuple(approx[start][0])
        end_point = tuple(approx[end][0])
        far_point = tuple(approx[far][0])
        if distance(start_point, far_point) > MIN_DISTANCE and distance(end_point, far_point) > MIN_DISTANCE:
            minutiae.append((start_point, end_point, far_point))

    return minutiae



def extract_minutiae_path(img_path):
    # Open the image and convert to grayscale
    img = Image.open(img_path).convert("L")

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Perform binarization using Otsu's method
    _, img_bin = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply a median filter to remove noise
    img_filt = cv2.medianBlur(img_bin, 5)

    # Apply a thinning algorithm to extract the ridge skeleton
    img_thin = thinning(img_filt)

    # Extract minutiae from the thinned image
    minutiae = extract_minutiae_from_skeleton(img_thin)

    return minutiae




def match_fingerprints_path(minutiae1, minutiae2):
    # Find corresponding minutiae pairs
    correspondences = []
    print("----", minutiae1)
    for m1 in minutiae1:
        for m2 in minutiae2:
            print("----", m1[2])
            if m1[2] == m2[2]:  # Same type (ending or bifurcation)
                dist = np.sqrt((m1[0]-m2[0])**2 + (m1[1]-m2[1])**2)
                print("----", dist)
                if dist <= 20:  # Maximum distance threshold
                    correspondences.append((m1, m2, dist))

    # Compute the score as the number of correspondences
    score = len(correspondences)
    return score



# img1_ = cv2.imread("enhanced/102_3.tif", cv2.IMREAD_GRAYSCALE)
# img2_ = cv2.imread("enhanced/101_1.tif", cv2.IMREAD_GRAYSCALE)

# img1 = enhance_function.enhance_fingerprint_2(img1_)
# img2 = enhance_function.enhance_fingerprint_2(img2_)


minutiae1 = extract_minutiae_path('enhanced/102_3.tif')
minutiae2 = extract_minutiae_path('enhanced/102_3.tif')


socre = match_fingerprints_path(minutiae1, minutiae2)
print("----------------", socre)
