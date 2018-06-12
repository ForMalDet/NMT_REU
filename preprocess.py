#!/usr/bin/python3
import cv2
import numpy as np

# Calculate width based on pdf length
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

# Create the pdf image
def create_image(filename, display=False):
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

    # Display image
    if display:
        cv2.imshow('image', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return img
