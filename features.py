#!/usr/bin/python3
import cv2
import numpy as np
import progressbar as pb

import ui

from skimage.feature import local_binary_pattern

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
 
    def describe(self, image, eps=1e-7):
        # compute the LBP representation to build the histogram of patterns
        lbp = local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))
 
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
 
        # return the histogram of Local Binary Patterns
        return hist

def describe_keypoints(img, alg, vector_size, descriptor_size, display):
    # Finding image keypoints
    kps = alg.detect(img)

    # Get first sorted <vector_size> points.
    kps = sorted(kps, key=lambda x: x.response)[:vector_size]
    kps, dsc = alg.compute(img, kps)

    # Flatten and normalize descriptors
    dsc = dsc.flatten()
    dsc = np.divide(dsc, 256)

    # optional Display image
    if display:
        img=cv2.drawKeypoints(img, kps, None)
        cv2.imshow('image', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return dsc

# Feature extractor
def extract_features(images, vector_size=32):
    # options = ["ORB", "SIFT", "KAZE", "LBP"]
    options = ["LBP"]
    res = ui.prompt("Choose a feature selection algorithm:", options)
    type = options[int(res)]

    data = []
    for img in pb.progressbar(images): # Process each image
        if type == "ORB":              # Corner features
            alg = cv2.ORB_create() 
            descriptor_size = 32
            data.append(describe_keypoints(img, alg, vector_size, descriptor_size, display))
        elif type == "SIFT":           # Corner features (patented)
            alg = cv2.xfeatures2d.SIFT_create() 
            descriptor_size = 128
            data.append(describe_keypoints(img, alg, vector_size, descriptor_size, display))
        elif type == "KAZE":           # Corner features (patented)
            alg = cv2.KAZE_create()
            descriptor_size = 32
            data.append(describe_keypoints(img, alg, vector_size, descriptor_size, display))
        elif type == "LBP":            # Simple texture recognition
            alg = LocalBinaryPatterns(24, 8)
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            data.append(alg.describe(grey))
        else:
            print("ERROR: Type " + type + " not found (features.extract_features())\n")
            return 1

    return data
