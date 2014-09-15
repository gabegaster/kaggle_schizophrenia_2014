from make_benchmarks import *

def tune(data,labels, clf=None):
    clf = Pipeline([('num_features', 
               SelectKBest(f_classif,k=100)),
                    ('svm', svm.SVC(C=.01,kernel='rbf'))])
    param_grid = {
        'num_features__k':range(40,300),
        # 'svm__C':10.**np.arange(-3,4)
    }
    grid_search = GridSearchCV(clf, 
                               param_grid,
                               cv=5,
                               scoring="roc_auc",
                               n_jobs=-1)
    grid_search.fit(data,labels)
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for p in param_grid.keys():
        print p, best_parameters[p]

    plot_cs(grid_search)

    return grid_search

def plot_cs(grid_search):
    for name,params in grid_search.param_grid.items():
        plt.plot(params,
                 [c.mean_validation_score 
                  for c in grid_search.grid_scores_], 
                 label="validation score")
        plt.xticks(params[::len(params)/6])
        plt.xlabel(name)
        plt.xlim(min(params),max(params))
        plt.ylabel("Accuracy")
        plt.legend(loc='best')
        title = ( get_name(grid_search.best_estimator_) +
                  name.split("__")[-1] )
        plt.title(title)
        plt.savefig("%s.png"%(title))

def evaluate(data,labels, num_trials=100):
    header = ("name","ROC_score","var","max")
    df = pandas.DataFrame()
