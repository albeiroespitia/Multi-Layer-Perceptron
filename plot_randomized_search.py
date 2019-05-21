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
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, r2_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.utils.class_weight import compute_class_weight

from sklearn.model_selection import KFold

from sklearn.preprocessing import RobustScaler

from sklearn.multiclass	import OneVsRestClassifier




import pandas as pd


# get some data
data = pd.read_csv("Absenteeism_at_work.csv")
targetData = np.array(data[["Absenteeism time in hours"]])
#data = data.replace(0,np.NaN)
#data.fillna(data.mean(), inplace=True)

#print(data.describe(include='all'))
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
y = y.astype(int)


#le = preprocessing.LabelEncoder()
#y = y.apply(le.fit_transform)
#lb = preprocessing.LabelBinarizer()
#le.fit_transform(y)

# build a classifier
#clf = RandomForestClassifier()
#svr_regressor = SVR()
#mlp = MLPClassifier()
#svc = SVC()
#lr = LogisticRegression(penalty='l2')


#onev = OneVsRestClassifier(estimator=lr)


#print(np.unique(y[0::,0]))


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
#              "solver": ["lbfgs","saga"],
#              "max_iter": [380,400,480],
#              "multi_class": ["multinomial"],
#              "class_weight":[{0:1,1:1,2:0.9},{0:1,1:2,2:1},{0:1,1:3,2:3},{0:1,1:4,2:4}],
#              "n_jobs":[-1]}


#class_weights2 = compute_class_weight('balanced', np.unique(y), y)
#print(np.unique(y))
unique, counts = np.unique(y, return_counts=True)
#print(dict(zip(unique, counts)))
#print(y)

#Random Forest 50%
#param = {"max_depth": [3],
#              "max_features": [15,16],
#              "min_samples_split": [2,4,8,10,13,14],
#              "bootstrap": [True, False],
#              "criterion": ["entropy"],
#              "n_estimators":[6],
              #"class_weight":[
              #                  {0:0.9,1:1,2:2,3:1,4:1,5:2,8:2.1,40:2.1,7:2},
              #                  {0:0.9,1:1,2:2,3:1,4:1,5:2,8:2.1,40:2.1},
              #                  {0:0.9,1:1,2:2,3:1,4:1,5:2,8:2.1,40:2.1},
              #                  {0:0.9,1:1,2:2,3:1,4:1,5:2,8:2.1,40:2.1}
              #               ],
#              "class_weight": [None],
              #"class_weight":[{7:0.2}],
#              "min_weight_fraction_leaf":[0.0001],
#              "max_leaf_nodes": [10],
#              "min_impurity_decrease": [0],
#              "n_jobs":[-1]}

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






# Scaling..
#scaler = RobustScaler()
#X = scaler.fit_transform(X)

#scaler = StandardScaler()  



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scaler.fit(X_train)
#X_train = scaler.transform(X_train)  
#X_test = scaler.transform(X_test)
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.fit_transform(X_test)


#clf.fit(X_train,y_train)
#svr_regressor.fit(X,y.ravel())
#wea.fit(X_train,y_train)

#rid.fit(X_train,y_train)








#param = {
#"alpha_1":[0.2,0.1,0.0001],
#"copy_X":[True,False], 
#"fit_intercept":[True,False], 
#"n_iter":[1000,5000,100,100000],
#"normalize":[False,True], 
#"positive":[False,True], 
#"precompute":[False,True], 
#"random_state":[1,2,3],
#"selection":['cyclic'], 
#"tol":[0.0001], 
#"warm_start":[False]
#}

#RD_model = RidgeCV()

#param = {'alpha':[0.2,0.1,0.0001],
#                    'fit_intercept': [True,False], 
#                    'normalize' :[False, True],
#                    "max_iter":[1000,5000,100,100000],}



#y_pred = regr.predict(X_test)



from sklearn.linear_model import LinearRegression


from sklearn.model_selection import cross_val_score, cross_val_predict

#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=3, max_features=[14,15,16], max_leaf_nodes=None,
#            min_impurity_split=1e-07, min_samples_leaf=1,
#            min_samples_split=11, min_weight_fraction_leaf=0.0,
#            n_estimators=6, n_jobs=2, oob_score=False, random_state=0,
#            verbose=0, warm_start=False)



#scaler = StandardScaler()  
# Don't cheat - fit only on training data
#scaler.fit(X_train)  
#X_train = scaler.transform(X_train)  
# apply same transformation to test data
#X_test = scaler.transform(X_test)  


#scaler = RobustScaler()
#X = scaler.fit_transform(X)

#scaler = StandardScaler()  

#scaler.fit(X_train)
#X_train = scaler.transform(X_train)  
#X_test = scaler.transform(X_test)
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.fit_transform(X_test)



#MLPRegression 42%
param = {'solver': ['adam'],
        'max_iter': [800,830,850],
        'random_state':[2],
        'hidden_layer_sizes':[55,60,65,80],
        'alpha': [0.001],
        'learning_rate_init' : [0.1,0.001]}


cv_method = KFold(n_splits=4, shuffle=False)


from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score



grid_search = GridSearchCV(MLPRegressor(), param, n_jobs=-1, cv = cv_method,  error_score='raise')
grid_search.fit(X_train,y_train)
y_pred = grid_search.predict(X_test)


print(y_pred)
print(y_test)
print("score: ",grid_search.best_score_)
print("explained_variance_score (1.0)",explained_variance_score(y_test, y_pred))
print("max_error (0.0)",max_error(y_test, y_pred))
print("mean_absolute_error (0.0) ",mean_absolute_error(y_test, y_pred))
print("mean_squared_error (0.0)",mean_squared_error(y_test, y_pred))
#print("mean_squared_log_error (0.0)",mean_squared_log_error(y_test, y_pred))
print("median_absolute_error (0.0)",median_absolute_error(y_test, y_pred))
print("r2_score (1.0)",r2_score(y_test, y_pred))





#result = cross_val_score(regr, X_train, y_train, cv=cv_method, scoring='accuracy')
#2print(result.mean())


#print(grid_search.cv_results_)


# The coefficients
#print('Coefficients: \n', grid_search.coef_)
# The mean squared error
#print("Mean squared error: %.2f"
#      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(y_test, y_pred))

#print("Accuracy score",accuracy_score(y_test, y_pred))
#print("accuracy",accuracy_score())


#print(X_test)
#print(y_pred)
#print(y_test)



#print("Best score: ",grid_search.best_score_)



#print("CLF: ",clf.score(X_train,y_train))
#print("GRID_SEARCHING: ",grid_search.score(X_train,y_train))

#print('R2: ', r2_score(y_pred = grid_search.best_estimator_.predict(X), y_true = y))


#grid_search = GridSearchCV(regr,param_grid=param_grid, cv=4, iid=False)
#print(grid_search.get_params())
start = time()
#print(grid_search.best_estimator_)
#print(grid_search.score(X_test,y_test))
#grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
