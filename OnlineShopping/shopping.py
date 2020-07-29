import csv
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

TEST_SIZE = 0.4

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).
    """
    # read the csv file and convert it to a pandas DataFrame
    dataset = pd.read_csv(filename)
    # get the numbers of rows
    row_num = len(dataset)
    # separate x and y matrix
    evidence = dataset.iloc[:, :-1].values
    labels = dataset.iloc[:, -1].values
    # create a month dict containing the numeric values for each month abbreviation
    months = dict(
        Jan=0,
        Feb=1,
        Mar=2,
        Apr=3,
        May=4,
        June=5,
        Jul=6,
        Aug=7,
        Sep=8,
        Oct=9,
        Nov=10,
        Dec=11
    )
    # encode the months on the 10th column to corresponding ints
    for i in range(row_num):
        evidence[i, 10] = months[evidence[i, 10]]
    # encode visitortype into integers
    evidence[:, -2] = (evidence[:, -2] == "Returning_Visitor").astype(int)
    # encode weekend into integers
    evidence[:, -1] = (evidence[:, -1] == True).astype(int)
    # encode revenue into into integers
    labels = (labels == True).astype(int)

    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(evidence, labels)

    return classifier


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple of (sensitivity, specificty).

    `sensitivity`: true positive rate
    `specificity`: true negative rate
    """
    cm = confusion_matrix(labels, predictions)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fp)
    specificity = tn / (tn + fn)

    return sensitivity, specificity


if __name__ == "__main__":
    main()
