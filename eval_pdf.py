#!/usr/bin/python3
import sys
import os
import gc
import cv2
import numpy as np
import progressbar as pb

import preprocess as pproc
import features as ft
import trainer as tr

def clear():
    """Clear screen, return cursor to top left"""
    sys.stdout.write('\033[2J')
    sys.stdout.write('\033[H')
    sys.stdout.flush()

def showUsage():
    print("Usage: ./eval_pdf.py <directory>\n")

def prompt(msg="Select an option:", options=[]):
    while True:
        print("\n" + msg)
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
    gc.collect()
    clear()

    if len(sys.argv) < 2:
        return showUsage()
    
    # Pre-process PDFs
    files = []
    images = []
    targets = []
    dirname = sys.argv[1]
    showImage = len(os.listdir(dirname)) <= 4 # Only show images for small data sets
    options = ["Load", "Create"]
    res = prompt("Load pre-processed images or create new ones?", options)
    filetype = ".bmp" if res == "0" else ".pdf"
    for file in pb.progressbar(os.listdir(dirname)):
        if file.endswith(filetype):
            files.append(os.path.join(dirname, file))
            if filetype == ".bmp":
                images.append(cv2.imread(files[-1]))
                targets.append(file[:5]) # Either "CLEAN" or "INFEC"
            elif filetype == ".pdf":
                images.append(pproc.create_image(files[-1], display=showImage))
                cv2.imwrite("{}.bmp".format(files[-1]), images[-1])
                targets.append(file[:5]) # Either "CLEAN" or "INFEC"

    # Extract feature vector
    options = ["ORB", "SIFT", "KAZE", "LBP"]
    res = prompt("Choose a feature selection algorithm:", options)
    data = []
    for img in pb.progressbar(images):
        data.append(ft.extract_features(img, type=options[int(res)], display=showImage))

    # Create and train model
    rows = []
    columns = []
    for row in range(len(images)):
        rows.append(row)
    for col in range(len(data[0])):
        columns.append(col)
    tr.train(data, rows, columns, targets)

if __name__ == "__main__":
    main()
