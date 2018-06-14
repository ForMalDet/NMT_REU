#!/usr/bin/python3
import sys
import gc

import preprocess as pproc
import features as ft
import trainer as tr
import ui

def showUsage():
    print("Usage: ./main.py <directory>\n")

def main():
    gc.collect() # Garbage collect

    # Check arguments
    if len(sys.argv) < 2:
        return showUsage()
    ui.clear()
    
    # Pre-process PDFs
    dirname = sys.argv[1]
    images, targets = pproc.processPDFs(dirname)

    # Extract feature vectors
    data = ft.extract_features(images)

    # Create, train, and evaluate model (until user quits)
    done = False
    while not done:
        tr.train(data, targets)
        options = ["Try another model", "Quit"]
        res = ui.prompt(options=options)
        if res == "1":
            done = True

if __name__ == "__main__":
    main()
