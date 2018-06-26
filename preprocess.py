import sys # Command line arguments
import os # Makes directories and detects filesize
import numpy as np # Makes pixel array
import math # Aids "length" variable
import scipy.misc as smp # Required to export original png
import skimage.feature.texture as txt # Required to export texture png
import multiprocessing as mp # PARALELL PROCESSING
import time
import ui # For prompts

pagesize = 4096

def makebitarray(files):
    pdfs = []
    print("converting pdfs to byte arrays...")
    for name in sys.argv:
        with open(name, "rb") as f:
            pdfs.append([name, list(f.read())])

    return pdfs

def getdims(f):
    """ Processes image with, height, and size.
    @param f is the file.
    @return (width, height, filesize(bytes).
    """
    size = os.path.getsize(f) * .001
    # width
    # TODO: make this prettier
    if (size <= 10):
        width = 32
    elif (10 < size <= 30):
        width = 64
    elif (30 < size <= 60):
        width = 128
    elif (60 < size <= 100):
        width = 256
    elif (100 < size <= 200):
        width = 384
    elif (200 < size <= 500):
        width = 512
    elif (500 < size <= 1000):
        width = 768
    else:
        width = 1024

    # height
    height = math.ceil(size*1000 // width) + 1
    return (width, height, size*1000)

def preprocess(pdf):
    dimensions = getdims(pdf[0])
    
    # Creating Image-----------------------------------------------------------
    data = np.array(pdf[1])
    data = np.pad(
        data, (0, dimensions[0]-(len(data)%dimensions[0])), 'constant')
    data = np.reshape(data, (-1, dimensions[0]))

    # Saving Image-------------------------------------------------------------
    img = smp.toimage(data)
    smp.imsave(pdf[0]+"(original).png", img)


def mode1(args):
    sys.argv = sys.argv[1:] # Stripping the name of this file -----------------

    start = time.time()

    # If argument was a directory and not the files in the directory
    # Ex: /root/thing/ instead of /root/thing/*
    if len(sys.argv) == 1 and sys.argv[0][-1] == "/": 
        directory = sys.argv[0].split("/")[0]
        sys.argv = [directory + "/" + item for item in os.listdir(sys.argv[0])]

    pdfs = makebitarray(sys.argv)

    # Parallel processing the image creation process. -------------------------
    print("Creating images...")
    pool = mp.Pool(mp.cpu_count()-1)
    if(len(pdfs) > 1000): # Have to break up, otherwise too much RAM
        tmp = 0
        for i in range(1000, len(pdfs), 1000):
            pool.map(preprocess, pdfs[tmp:i])
            tmp = i
        pool.map(preprocess, pdfs[tmp:])
    else:
        pool.map(preprocess, pdfs)

    # Printing time -----------------------------------------------------------
    end = time.time()
    print("\nConverted {0} files in {1} seconds"
        .format(len(sys.argv), end-start ))

    # Return Values -----------------------------------------------------------
    images = [np.array(item[1]) for item in pdfs]
    print(type(images))
    targets = [item[0] for item in pdfs]
    return images, targets


def mode2(args):
    pass

def processPDFs(sys):

    """
    if not len(sys.argv):
        print("At least 1 pdf filename required")
        quit()
    """

    options = ["Load", "Create"]
    mode = ui.prompt("Load pre-processed images or create new ones?", options)

    if int(mode):
        return mode1(sys.argv)
    elif not int(mode):
        mode2(sys.argv)

if __name__ == "__main__":
    processPDFs()
