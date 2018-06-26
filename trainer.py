#!/usr/bin/python3
import numpy as np
import cv2

from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_score

import ui

def train(data, targets):
    options = ["Support Vector Machine", "Random Forest",
            "Decision Tree Classifier", "KNN"]
    res = ui.prompt("Choose a ML algorithm:", options)
    switch = {
        0: svm.SVC(C=100.),
        1: RandomForestClassifier(max_depth=2),
        2: DecisionTreeClassifier(),
        3: KNeighborsClassifier()
    }
    clf = switch.get(int(res))

    # Split-out test dataset and randomize order
    seed = 42 # Seeded so as to create reproducible results
    validation_size = 0.20
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
            data, targets, test_size=validation_size, random_state=seed
            )

    # Fit model
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Show confusion matrix for this fit
    unique, counts = np.unique(y_pred, return_counts=True)
    counts = dict(zip(unique, counts))
    print("\nPredictions: {}".format(counts))
    print("Confusion Matrix:\n{}".format(confusion_matrix(y_test, y_pred)))

    # Cross validate and calculate scores
    scoring = ["accuracy", "precision", "recall", "f1"] # Choose scoring methods
    targets = [val == "INFEC" for val in targets] # Set INFEC as positive val
    scores = cross_validate(clf, data, targets, scoring=scoring, cv=5)
    print("Scores calculated from 5-fold cross validation:")
    print("Accuracy:  {},\t{}".format(round(np.mean(scores["test_accuracy"]),  4), scores["test_accuracy"]))
    print("Precision: {},\t{}".format(round(np.mean(scores["test_precision"]), 4), scores["test_precision"]))
    print("Recall:    {},\t{}".format(round(np.mean(scores["test_recall"]),    4), scores["test_recall"]))
    print("F1:        {},\t{}".format(round(np.mean(scores["test_f1"]),        4), scores["test_f1"]))
