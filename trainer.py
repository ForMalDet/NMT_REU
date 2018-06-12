#!/usr/bin/python3
import numpy as np
import cv2

import matplotlib.pyplot as plt

import pandas
from pandas.plotting import scatter_matrix

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def train(data, rows, columns):
    # Create DataFrame
    data = np.array(data).reshape((len(rows),len(columns)))
    df = pandas.DataFrame(data,index=rows,columns=columns)
    print(df)
