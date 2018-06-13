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
import ui

def showUsage():
    print("Usage: ./eval_pdf.py <directory>\n")

def main():
    gc.collect() # Garbage collect

    # Check arguments
    if len(sys.argv) < 2:
        return showUsage()
    ui.clear()
    
    # Pre-process PDFs
    dirname = sys.argv[1]
    images, targets = pproc.processPDFs(dirname)

    # Extract feature vector
    options = ["ORB", "SIFT", "KAZE", "LBP"]
    res = ui.prompt("Choose a feature selection algorithm:", options)
    data = []
    for img in pb.progressbar(images):
        data.append(ft.extract_features(img, type=options[int(res)], display=False))

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
