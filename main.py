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

    # Extract feature vectors (until user quits)
    doneExtracting = False
    while not doneExtracting:
        data = ft.extract_features(images)

        # Create, train, and evaluate model (until user quits)
        doneTraining = False
        while not doneTraining:
            tr.train(data, targets)
            options = ["Try another model", "Extract new features", "Quit"]
            res = ui.prompt(options=options)
            if options[int(res)] == "Quit":
                doneTraining = True
                doneExtracting = True
            elif options[int(res)] == "Extract new features":
                doneTraining = True
        gc.collect() # Garbage collect

if __name__ == "__main__":
    main()
