'''Goes through a bunch of possible classification pipelines and
scores each with cross-validation and records scores.
'''
import numpy as np
import csv
from itertools import izip

# classifiers / transformers / pipelines
from sklearn import naive_bayes
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.lda import LDA
# from sklearn.qda import QDA
from sklearn.feature_selection import RFE
from sklearn.neural_network import BernoulliRBM
from sklearn import decomposition

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection.univariate_selection import \
    SelectKBest, f_classif
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV

from matplotlib import pyplot as plt
import pandas

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

def bootstrap(data, labels, clf=naive_bayes.BernoulliNB):
    sample = np.random.sample(len(data)) < .75
    clf.fit(data[sample], labels[sample])
    return roc_score(clf,data[-sample],labels[-sample])

def roc_score(clf, data, labels):
    predictions = get_score(clf, data)
    return metrics.roc_auc_score(labels, predictions)

def get_score(clf, data):
    try:
        out = clf.decision_function(data).ravel()
    except AttributeError:
        try:
            out = clf.predict_proba(data)[:,1]
        except AttributeError:
            out = clf.predict(data)
    return out
    
def write_predictions(clf):
    ids, data = load_data(False)
    # preds = clf.predict_proba(data)[:,1]
    preds = get_score(clf, data)

    with open("submissions/" + get_name(clf)+".csv",'w') as f:
        w = csv.writer(f)
        w.writerow(["ID","Probability"])
        for item in izip(ids, preds):
            w.writerow(item)

def get_name(thing):
    if hasattr(thing, "steps"):
        return "_".join([i[0] for i in thing.steps])
    else:
        return thing.__repr__().split("(")[0]

def stringify(i):
    return "%.3f"%i

def evaluate(data,labels, num_trials=100):
    header = ("name","ROC_score","var","max")
    df = pandas.DataFrame()

    for c in CLASSIFIERS:
        scores = np.array([bootstrap(data, labels, clf=c)
                           for i in xrange(num_trials)])
        row = dict(zip(header, [get_name(c),
                                scores.mean(),
                                scores.var(), 
                                scores.max(),]))
        row = pandas.DataFrame([row])
        df = df.append(row)
    df.index = df.name
    df = df.drop("name",1)
    df = df.sort("ROC_score",ascending=False)
    print df.to_string(float_format=stringify)
    return df

CLASSIFIERS = [
    svm.SVC(), 
    LogisticRegression(C=0.16,penalty='l1', ## given in the forums
                       tol=0.001, fit_intercept=True)
    Pipeline([('pca',decomposition.PCA()),
              ('svm',svm.SVC()),]),
    Pipeline([("rfe_Lsvc",
               RFE(estimator=svm.LinearSVC(), 
                   n_features_to_select=240,step=1)),
              ("svc_3",svm.SVC(gamma=.1,
                             degree=3,
                             kernel="rbf",
                             C=10)),]),
    Pipeline([("rfe_Lsvc",
               RFE(estimator=svm.LinearSVC(), 
                   n_features_to_select=240,step=1)),
              ("svc_5",svm.SVC(C=1000,
                             gamma=.1,
                             degree=5,
                             kernel="rbf")),]),
    Pipeline([("rfe_Lsvc",
               RFE(estimator=svm.LinearSVC(), 
                   n_features_to_select=282,step=1)),
              ("svc",svm.SVC()),]),
    Pipeline([("85_best",SelectKBest(k=100)),
              ("svc",svm.SVC(C=.01)),]),
    Pipeline([("normalize", StandardScaler()),
              ("grid_search_svm", GridSearchCV(
                  svm.SVC(), {
                      'C': 10**np.arange(5),
                      'gamma': [0, 1e-5, 1e-3, 1e-1,],
                      'kernel': ['linear','rbf'],
                      "degree":range(1,10),
                  },
                  cv=5,
                  scoring="roc_auc",
                  n_jobs=-1))]),
]

if __name__ == "__main__":
    ids, data, labels = load_data()
    #gs = tune(data,labels)
    evaluate(data,labels)
    # clf = CLASSIFIERS[-1]
    # write_predictions(clf.fit(data,labels))

def inspect_num_features():
    the_mean = []
    the_vars = []
    X = range(30,400,10)
    for k_best in X:
        p = Pipeline([('f_test_%s_best'%k_best, SelectKBest(f_classif,k=k_best)),
                      ('naive_bayes', naive_bayes.BernoulliNB())])
        scores = np.array([bootstrap(data,labels,clf=p) for i in xrange(300)])
        the_mean.append(scores.mean())
        the_vars.append(scores.var())

    the_vars = np.array(the_vars)
    the_mean = np.array(the_mean)

    plt.plot(X, the_mean, color='r')
    plt.plot(X, the_mean + 2*the_vars, X, the_mean - 2*the_vars, color='b')
    plt.show()

OTHER_CLASSIFEIRS_TO_TRY = [
    naive_bayes.GaussianNB(),
    naive_bayes.BernoulliNB(), 
    Pipeline([("normalize", StandardScaler()),
              ("svm",svm.SVC()),]),
    svm.LinearSVC(), 
    RandomForestClassifier(),
    AdaBoostClassifier(),
    # # LinearRegression(),
    LogisticRegression(),
    GIVEN_OPT,
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    Pipeline([('linear_svm',svm.LinearSVC()),
              ('bernoulli',naive_bayes.BernoulliNB())]),
    Pipeline([("rfe_svm",
               RFE(estimator=svm.SVC(kernel="linear"), 
                   n_features_to_select=1, step=320)),# new
              ('bernoulli',naive_bayes.BernoulliNB())]),
    Pipeline([('f_test_100_best', 
               SelectKBest(f_classif,k=100)),
              ('naive_bayes', naive_bayes.BernoulliNB())]),
    Pipeline([("normalize", StandardScaler()),
              ('f_test_20_best', SelectKBest(f_classif,k=20)),
              ("grid_search_svm", GridSearchCV(
                  svm.SVC(), {
                      'C': 10**np.arange(5),
                      'gamma': [0, 1e-5, 1e-3, 1e-1,],
                      'kernel': ['linear','rbf'],
                      'degree': range(1,5),
                  },
                  cv=5,
                  scoring="roc_auc",
                  n_jobs=-1))]),
    Pipeline([("normalize", StandardScaler()),
              ('f_test_100_best',SelectKBest(f_classif,k=100)),
              ("grid_search_svm", GridSearchCV(
                  svm.SVC(), {
                      'C': 10**np.arange(5),
                      'gamma': [0, 1e-5, 1e-3, 1e-1,],
                      'kernel': ['linear','rbf'],
                      'degree': range(1,5),
                  },
                  cv=5,
                  scoring="roc_auc",
                  n_jobs=-1))]),
    Pipeline([("normalize", StandardScaler()),
              ('f_test_200_best',SelectKBest(f_classif,k=200)),
              ("grid_search_svm", GridSearchCV(
                  svm.SVC(), {
                      'C': 10**np.arange(5),
                      'gamma': [0, 1e-5, 1e-3, 1e-1,],
                      'kernel': ['linear','rbf'],
                      'degree': range(1,5),
                  },
                  cv=5,
                  scoring="roc_auc",
                  n_jobs=-1))]),
    Pipeline([('rbm', BernoulliRBM(verbose=True)),
              ("logistic",LogisticRegression())]),
    Pipeline([('rbm', BernoulliRBM(verbose=True)),
              ("bernoulli",naive_bayes.BernoulliNB())]),
    Pipeline([("normalize", StandardScaler()),
              #("ftest_100_best",SelectKBest(f_classif,k=100)),
              ("log_reg_opt",GIVEN_OPT)]),
]
