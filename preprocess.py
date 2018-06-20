#!/usr/bin/python3
import sys
import os
import cv2
import numpy as np
import progressbar as pb
from math import isinf

import ui

# Calculate width based on pdf length
# Return: width
def image_width(length):
    width = 32
    if length < 10000:
        width = 32
    elif length < 30000:
        width = 64
    elif length < 60000:
        width = 128
    elif length < 100000:
        width = 256
    elif length < 200000:
        width = 384
    elif length < 500000:
        width = 512
    elif length < 1000000:
        width = 768
    else:
        width = 1024
    return width

# Create the BMP pdf image
# Return: img
def create_bmp(filename):
    pdf = open(filename, "rb").read()

    # Calculate height and width of image
    width = image_width(len(pdf))
    height = int(len(pdf) / width) + 1

    # Initialize image
    img = np.zeros((height, width, 3), np.uint8)

    # Fill image
    for row in range(height):
        for col in range(width):
            pos = row * width + col
            val = 0
            if pos < int(len(pdf)): # Only get val if pos is within pdf length
                val = pdf[row * width + col]
            img[row, col] = [val]

    return img

# Create the Markov Plot visualization of pdf
# Return: img
def create_markov(filename):
    pdf = open(filename, "rb").read()
    p = np.zeros(shape=(256, 256))
    for i in range(len(pdf)-1):
        row = pdf[i]
        col = pdf[i+1]
        p[row, col] += 1

    for row in range(256):
        sum = np.sum(p[row])
        if sum != 0:
            p[row] /= sum

    # Normalize
    #p += 0.00000000000000001
    #p = np.divide(1, p)
    p = (1 / np.ndarray.max(p)) * p
    #p = np.subtract(1, p)
    p *= 255

    # Convert to RGB color spectrum
    img = np.zeros(shape=(256, 256, 3))
    for row in range(256):
        for col in range(256):
            val = p[row, col]
            val = val if not isinf(val) else 0
    #        if val < 256:
    #            img[row, col] = np.array([255-val, val, 0])
            img[row, col] = [val, val, val]
    #        else:
    #            img[row, col] = np.array([0, 511-val, val-256])
    
    return img.astype(np.uint8)

# Open PDFs or BMPs, creating images if prompted to
# Return: (images[], targets[])
def processPDFs(dirname):
    images = []
    targets = []

    # Choose to load old images or create new ones
    options = ["Load", "Create"]
    res = ui.prompt("Load pre-processed images or create new ones?", options)
    filetype = ".bmp" if res == "0" else ".pdf"

    # If creating new ones, select a type
    type = None
    if options[int(res)] == "Create":
        options = ["Byte Map", "Markov Plot"]
        res = ui.prompt(options=options)
        type = options[int(res)]

    for file in pb.progressbar(os.listdir(dirname)): # Iterate through files
        if file.endswith(filetype):                  # Check if right file type
            filepath = os.path.join(dirname, file)
            if filetype == ".bmp":                   # We are just loading an image here
                images.append(cv2.imread(filepath))
                targets.append(file[:5])             # Either "CLEAN" or "INFEC"
            elif filetype == ".pdf":                 # Creating new images here
                if type == "Byte Map":
                    images.append(create_bmp(filepath))
                elif type == "Markov Plot":
                    images.append(create_markov(filepath))
                cv2.imwrite("{}.bmp".format(filepath), images[-1])
                targets.append(file[:5])            # Either "CLEAN" or "INFEC"

    return images, targets
