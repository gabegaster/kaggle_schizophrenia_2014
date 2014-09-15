from make_benchmarks import *

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
