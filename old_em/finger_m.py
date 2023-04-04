import numpy as np
import math

# Function to calculate Euclidean distance between two points
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Function to pair FeaturesTerminations and FeaturesBifurcations
def pair_features(features_1, features_2):
    matched_terminations = []
    matched_bifurcations = []
    for termination_1 in features_1["Terminations"]:
        min_distance = float('inf')
        matched_termination = None
        for termination_2 in features_2["Terminations"]:
            distance = euclidean_distance(termination_1[0:2], termination_2[0:2])
            if distance < min_distance:
                min_distance = distance
                matched_termination = termination_2
        if matched_termination is not None:
            matched_terminations.append((termination_1, matched_termination))
    for bifurcation_1 in features_1["Bifurcations"]:
        min_distance = float('inf')
        matched_bifurcation = None
        for bifurcation_2 in features_2["Bifurcations"]:
            distance = euclidean_distance(bifurcation_1[0:2], bifurcation_2[0:2])
            if distance < min_distance:
                min_distance = distance
                matched_bifurcation = bifurcation_2
        if matched_bifurcation is not None:
            matched_bifurcations.append((bifurcation_1, matched_bifurcation))
    return matched_terminations, matched_bifurcations

# Function to calculate pairwise distance between matched terminations and bifurcations
def pairwise_distance(matched_terminations, matched_bifurcations):
    distances = []
    for termination_pair in matched_terminations:
        for bifurcation_pair in matched_bifurcations:
            distance = euclidean_distance(termination_pair[0][0:2], bifurcation_pair[0][0:2])
            distances.append(distance)
    return distances

# Function to perform fingerprint pairing
def fingerprint_matching(features_1, features_2, threshold):
    matched_terminations, matched_bifurcations = pair_features(features_1, features_2)
    distances = pairwise_distance(matched_terminations, matched_bifurcations)
    if len(distances) > 0:
        avg_distance = sum(distances) / len(distances)
        if avg_distance < threshold:
            return True
    return False

# Here, features_1 and features_2 are the features extracted from two fingerprints. The features should be in the format:
# features_1 = {"Terminations": [(LocX, LocY, theta), ...],
#               "Bifurcations": [(LocX, LocY, theta1, theta2, theta3), ...]}
# LocX and LocY are the coordinates of the termination or bifurcation point. theta is the angle of the ridge for terminations, and theta1, theta2, and theta3 are the angles of the three ridges for bifurcations.

# The function fingerprint_matching takes in threshold as a parameter, which is the maximum pairwise distance between the matched features for the fingerprints