'''Trains SupportVectorMachine and uses it to write predictions.
'''
from sklearn import svm

from utils import load_data, write_predictions

if __name__ == "__main__":
    ids, data, labels = load_data()
    clf = svm.SVC().fit(data,labels)
    write_predictions(clf)
