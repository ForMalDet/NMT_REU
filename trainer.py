#!/usr/bin/python3
import numpy as np
import cv2

from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

import ui

def train(data, targets):
    options = ["Support Vector Machine", "Random Forest"]
    res = ui.prompt("Choose a ML algorithm:", options)
    switch = {
        0: svm.SVC(C=100.),
        1: RandomForestClassifier(max_depth=2),
    }
    clf = switch.get(int(res))

    # Split-out test dataset and randomize order
    seed = 42 # Seeded so as to create reproducible results
    validation_size = 0.20
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data, targets, test_size=validation_size, random_state=seed)

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("\nAccuracy: {}".format(accuracy_score(y_test, y_pred)))
    print("\nConfusion Matrix:\n{}".format(confusion_matrix(y_test, y_pred)))
