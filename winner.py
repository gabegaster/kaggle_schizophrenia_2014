from make_benchmarks import load_data, write_predictions
from sklearn import svm

if __name__ == "__main__":
    ids, data, labels = load_data()
    clf = svm.SVC().fit(data,labels)
    write_predictions(clf)
