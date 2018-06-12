#!/usr/bin/python3
import cv2
import numpy as np

# Feature extractor
def extract_features(img, vector_size=32, type="ORB", display=False):
    alg = None
    descriptor_size = 0
    if type == "ORB":
        alg = cv2.ORB_create() 
        descriptor_size = 32
    elif type == "SIFT":
        alg = cv2.xfeatures2d.SIFT_create() 
        descriptor_size = 128
    elif type == "KAZE":
        alg = cv2.KAZE_create()
        descriptor_size = 32
    else:
        print("ERROR: Type " + type + " not found (features.extract_features())\n")
        return 1

    # Finding image keypoints
    kps = alg.detect(img)

    # Get first sorted <vector_size> points.
    kps = sorted(kps, key=lambda x: x.response)[:vector_size]
    kps, dsc = alg.compute(img, kps)

    # Flatten and normalize descriptors
    dsc = dsc.flatten()
    dsc = np.divide(dsc, 256)

    # Display image
    if display:
        img=cv2.drawKeypoints(img,kps,None)
        cv2.imshow('image', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return dsc
