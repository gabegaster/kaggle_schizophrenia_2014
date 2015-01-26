'''Helper utilities to load training / testing data and write predictions.
'''


import csv
from itertools import izip
import numpy as np

def load_data(train=True, sbm_only=False, fnc_only=False):
    if train:
        fnc="Train/train_FNC.csv"
        sbm="Train/train_SBM.csv"
    else:
        fnc="Test/test_FNC.csv"
        sbm="Test/test_SBM.csv"
        
    with open(fnc,'r') as f:
        train_fnc = list(csv.reader(f))
    fnc_header = train_fnc[0]
    fnc_data = np.array([np.array(map(float,i)) for i in train_fnc[1:]])
    ids = np.array(fnc_data[:,0],dtype=int)

    with open(sbm,'r') as f:
        train_sbm = list(csv.reader(f))
    sbm_header = train_sbm[0]
    sbm_data = np.array([np.array(map(float,i)) for i in train_sbm[1:]])
    fnc_data = fnc_data[:,1:]
    sbm_data = sbm_data[:,1:]
    data = np.column_stack((sbm_data,fnc_data))
    
    if not train:
        return ids, data

    with open("train/train_labels.csv",'r') as f:
        f.next()
        labels = np.array([int(i[1]) for i in csv.reader(f)])

    if sbm_only:
        return ids,sbm_data,labels
    elif fnc_only:
        return ids,fnc_data,labels
    else:
        return ids, data, labels

def write_predictions(clf):
    ids, data = load_data(False)
    # preds = clf.predict_proba(data)[:,1]
    preds = get_score(clf, data)

    with open("submissions/" + get_name(clf)+".csv",'w') as f:
        w = csv.writer(f)
        w.writerow(["ID","Probability"])
        for item in izip(ids, preds):
            w.writerow(item)

def get_score(clf, data):
    '''Allows several of different kinds of classifiers,
    interchangably. Some (like random forests, SVMs, and logistic
    regression) have the method decision_function and some (like naive
    bayes) have predict_proba.

    '''
    try:
        out = clf.decision_function(data).ravel()
    except AttributeError:
        try:
            out = clf.predict_proba(data)[:,1]
        except AttributeError:
            out = clf.predict(data)
    return out
    
