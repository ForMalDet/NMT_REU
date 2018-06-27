#!/usr/bin/python3
import cv2
import numpy as np
import progressbar as pb

import ui

from scipy import ndimage as nd
from skimage.util import img_as_float
from skimage import exposure

from skimage.feature import local_binary_pattern
from skimage.filters import gabor_kernel
from skimage.filters.rank import entropy
from skimage.morphology import disk

# Compute_feats for gabor filter
def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = nd.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats

# Create histogram of pixel values in an image
def histogram(image, nBins, range=None, eps=1e-7):
    (hist, _) = np.histogram(image.ravel(),
        bins=np.arange(0, nBins))

    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

    return hist

# Simple LBP algorithm 
class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
 
    def describe(self, image, eps=1e-7):
        # compute the LBP representation to build the histogram of patterns
        lbp = local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        hist = histogram(lbp, self.numPoints+3, self.numPoints+2, eps)
        return hist

# Create large description vector for keypoints in an image
def describe_keypoints(img, alg, vector_size, descriptor_size, display=False):
    # Finding image keypoints
    kps = alg.detect(img, None)

    # Get first sorted <vector_size> points.
    kps = sorted(kps, key=lambda x: x.response)[:vector_size]
    kps, dsc = alg.compute(img, kps)

    # Fill with zeros if no keypoints are found
    if len(kps) < vector_size:
        dsc = np.zeros(shape=(vector_size, descriptor_size))

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
    options = ["ORB", "SIFT", "LBP", "Gabor", "Entropy", "LBP and Entropy"]
    res = ui.prompt("Choose a feature selection algorithm:", options)
    type = options[int(res)]

    data = []
    for img in pb.progressbar(images): # Process each image
        if type == "ORB":              # Corner features
            alg = cv2.ORB_create() 
            descriptor_size = 32
            data.append(describe_keypoints(img, alg, vector_size, descriptor_size))
        elif type == "SIFT":           # Corner features (patented)
            alg = cv2.xfeatures2d.SIFT_create() 
            descriptor_size = 128
            data.append(describe_keypoints(img, alg, vector_size, descriptor_size))
        elif type == "LBP":            # Simple texture recognition
            alg = LocalBinaryPatterns(32, 16)
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            data.append(alg.describe(grey))
        elif type == "Gabor":
            # prepare filter bank kernels
            kernels = []
            for theta in range(4):
                theta = theta / 8. * np.pi
                for sigma in (1, 3):
                    for frequency in (0.05, 0.25):
                        kernel = np.real(gabor_kernel(frequency, theta=theta,
                                                      sigma_x=sigma, sigma_y=sigma))
                        kernels.append(kernel)

            shrink = (slice(0, None, 3), slice(0, None, 3))
            img_shrink= img_as_float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[shrink]

            feats = compute_feats(img_shrink, kernels).flatten()
            hist = exposure.histogram(img_shrink, nbins=16)
            data.append(np.append(feats, hist))
        elif type == "Entropy":
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grey = entropy(grey, disk(5))
            hist = exposure.histogram(grey, nbins=16)[0]
            data.append(hist)
        elif type == "LBP and Entropy":
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            alg = LocalBinaryPatterns(32, 16)
            entropy_grey = entropy(grey, disk(5))
            hist = exposure.histogram(entropy_grey, nbins=16)[0]
            data.append(np.append(alg.describe(grey), hist))
        else:
            print("ERROR: Type " + type + " not found (features.extract_features())\n")
            return 1

    return data, type
