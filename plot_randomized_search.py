"""
=========================================================================
Comparing randomized search and grid search for hyperparameter estimation
=========================================================================

Compare randomized search and grid search for optimizing hyperparameters of a
random forest.
All parameters that influence the learning are searched simultaneously
(except for the number of estimators, which poses a time / quality tradeoff).

The randomized search and the grid search explore exactly the same space of
parameters. The result in parameter settings is quite similar, while the run
time for randomized search is drastically lower.

The performance is slightly worse for the randomized search, though this
is most likely a noise effect and would not carry over to a held-out test set.

Note that in practice, one would not search over this many different parameters
simultaneously using grid search, but pick only the ones deemed most important.
"""
print(__doc__)

import numpy as np

from time import time
from scipy.stats import randint as sp_randint
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier



import pandas as pd


# get some data
data = pd.read_csv("Absenteeism_at_work.csv")
targetData = np.array(data[["Absenteeism time in hours"]])
#data = data.replace(0,np.NaN)
#data.fillna(data.mean(), inplace=True)

print(data.describe(include='all'))
#digits = load_digits()
#print(data.describe())
#X, y = digits.data, digits.target


X = data.values[:,0:20] # LR
#X = data.iloc[:, 1:20].values
#y = data.iloc[:,20:].values.reshape(-1,1)
#print(X,y)
y = data.values[:,20] # LR
#u = digits.target[0:740]
#y = pd.DataFrame(data[["Absenteeism time in hours"]])
#print(y)
#y = y.astype(int)

#le = preprocessing.LabelEncoder()
#y = y.apply(le.fit_transform)
#lb = preprocessing.LabelBinarizer()
#le.fit_transform(y)

# build a classifier
clf = RandomForestClassifier(n_estimators=6)
#svr_regressor = SVR()
#wea = MLPClassifier()
#svc = SVC()
#lr = LogisticRegression(penalty='l2')



# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# use a full grid over all parameters

#Logistic Regretion 42%
#param = {"C": [0.001,0.01],
#              "solver": ["lbfgs"],
#              "max_iter": [380,400,480],
#              "multi_class": ["multinomial"],
#              "class_weight":[{0:1,1:1,2:0.9},{0:1,1:2,2:1},{0:1,1:3,2:3},{0:1,1:4,2:4}],
#              "n_jobs":[-1]}


#Random Forest 48%
param = {"max_depth": [3],
              "max_features": [15,16],
              "min_samples_split": [13,14],
              "bootstrap": [True, False],
              "criterion": ["entropy"],
              "class_weight":[
                                {0:0.9,1:1,2:2,3:1,4:1,8:2.1,40:2},
                                {0:0.9,1:1,2:2,3:1,4:1,8:2.1,40:1},
                                {0:0.9,1:1,2:2,3:1,4:1,8:2.1,40:40},
                                {0:0.9,1:1,2:2,3:1,4:1,8:2.1,40:30}
                             ],
              #"class_weight": [None],
              "min_weight_fraction_leaf":[0.0001],
              "max_leaf_nodes": [10],
              "min_impurity_decrease": [0],
              "n_jobs":[-1]}

#SvrRegresor # -1%
#param = {"C": [0.001,0.01],
#              "gamma": ['auto','scale'],
#              "epsilon": [0.2],
#              "kernel": ['rbf'],
#              "max_iter":[10000,50000,100000]}

#MLPClassifier 42%
#param = {'solver': ['lbfgs'],
#        'max_iter': [400,480,500,520],
#        'random_state':[2],
#        'hidden_layer_sizes':[30],
#        'alpha': [0.01],
#        'learning_rate_init' : [0.1]}

#SVC 43%
#param = {"C": [0.001],
#        'max_iter': [480,500,550],
#        'kernel' : ['linear'],
#        'gamma': ['auto']}




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#clf.fit(X,y)
#svr_regressor.fit(X,y.ravel())
#wea.fit(X,y)


#scaler = StandardScaler()  
#scaler.fit(X_train)
#X_train = scaler.transform(X_train)  
#X_test = scaler.transform(X_test)
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.fit_transform(X_test)

#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=3, max_features=[14,15,16], max_leaf_nodes=None,
#            min_impurity_split=1e-07, min_samples_leaf=1,
#            min_samples_split=11, min_weight_fraction_leaf=0.0,
#            n_estimators=6, n_jobs=2, oob_score=False, random_state=0,
#            verbose=0, warm_start=False)


grid_search = GridSearchCV(clf, param, n_jobs=-1, cv = 4)
grid_search.fit(X,y)

#grid_search = GridSearchCV(clf,param_grid=param_grid, cv=4, iid=False)
#print(grid_search.get_params())
start = time()
#grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
