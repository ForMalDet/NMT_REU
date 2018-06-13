#!/usr/bin/python3
import sys
import os
import cv2
import numpy as np
import progressbar as pb

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
            img[row, col] = [val, val, val]

    return img

# Open PDFs or BMPs, creating images if prompted to
# Return: (images[], targets[])
def processPDFs(dirname):
    images = []
    targets = []

    options = ["Load", "Create"]
    res = ui.prompt("Load pre-processed images or create new ones?", options)

    filetype = ".bmp" if res == "0" else ".pdf"
    for file in pb.progressbar(os.listdir(dirname)): # Iterate through files
        if file.endswith(filetype):                  # Check if right file type
            filepath = os.path.join(dirname, file)
            if filetype == ".bmp":
                images.append(cv2.imread(filepath))
                targets.append(file[:5])             # Either "CLEAN" or "INFEC"
            elif filetype == ".pdf":
                images.append(create_bmp(filepath))
                cv2.imwrite("{}.bmp".format(filepath), images[-1])
                targets.append(file[:5])            # Either "CLEAN" or "INFEC"

    return images, targets
