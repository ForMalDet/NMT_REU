#!/usr/bin/python3
import numpy as np
import cv2

import matplotlib.pyplot as plt

import pandas
from pandas.plotting import scatter_matrix 
# from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
from sklearn import svm

def train(data, rows, columns, targets):
    # Create DataFrame
    # data = np.array(data).reshape((len(rows),len(columns)))
    # df = pandas.DataFrame(data,index=rows,columns=columns)
    #print(df)
    #df.hist()
    #plt.show()
    
    # Split-out test dataset and randomize order
    seed = 7
    validation_size = 0.20
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data, targets, test_size=validation_size, random_state=seed)

    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(accuracy_score(y_test, y_pred))
