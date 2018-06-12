#!/usr/bin/python3
import sys
import os
import cv2
import numpy as np
import progressbar as pb

import preprocess as pproc
import features as ft

def showUsage():
    print("Usage: ./eval_pdf.py <directory>\n")

def prompt(msg="Select an option:", options=[]):
    while True:
        print(msg)
        count = 0
        for option in options:
            print("\t{}) {}".format(count, option))
            count += 1
        res = input(" > ")
        if  len(options) == 0 or int(res) < len(options):
            return res
        else:
            print("Please select a number between 0 and {}".format(len(options)-1))

def main():
    # Pre-process PDFs
    if len(sys.argv) < 2:
        return showUsage()
    filenames = []
    images = []
    dirname = sys.argv[1]
    showImage = len(os.listdir(dirname)) <= 4 # Only show images for small data sets
    options = ["Load", "Create"]
    res = prompt("Load pre-processed images or create new ones?", options)
    filetype = ".bmp" if res == "0" else ".pdf"
    for file in pb.progressbar(os.listdir(dirname)):
        if file.endswith(filetype):
            filenames.append(os.path.join(dirname, file))
            if filetype == ".bmp":
                images.append(cv2.imread(filenames[-1]))
            elif filetype == ".pdf":
                images.append(pproc.create_image(filenames[-1], display=showImage))
                cv2.imwrite("{}.bmp".format(filenames[-1]), images[-1])

    # Extract feature vector
    options = ["ORB", "SIFT", "KAZE"]
    res = prompt("Choose a feature selection algorithm:", options)
    for img in pb.progressbar(images):
        fvector = ft.extract_features(img, type=options[int(res)], display=showImage)

if __name__ == "__main__":
    main()
