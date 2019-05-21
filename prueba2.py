import pandas as pd
from time import time
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from sklearn.preprocessing import minmax_scale
#from sklearn.preprocessing import MaxAbsScaler
#from sklearn.preprocessing import RobustScaler
#from sklearn.preprocessing import Normalizer
#from sklearn.preprocessing import QuantileTransformer
#from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score, classification_report,confusion_matrix, accuracy_score
import scikitplot as skplt
import matplotlib.pyplot as plt
from collections import Counter



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



namesColumn = ["ID", "Reason for absence","Month of absence", "Day of the week", "Seasons", "Transportation expense", "Distance from Residence to Work", "Service time", "Age", "Work load Average/day ", "Hit target", "Disciplinary failure", "Education", "Son", "Social drinker", "Social smoker", "Pet", "Weight","Height","Body mass index","Absenteeism time in hours"]


dataframe = pd.read_csv('Absenteeism_at_work.csv', names=namesColumn, sep=",",header=0)

X = dataframe.iloc[:, 0:20]

#print(X)


y = pd.DataFrame(dataframe["Absenteeism time in hours"])

#print(y.values.ravel())
print(Counter(y.values.ravel()))


print("Valores",np.unique(y))
#y = y['Absenteeism time in hours'].apply(str)
for i in range(y.size):
    if(y.values[i] > 0 and y.values[i] <= 5):
         y.values[i] = 0
    elif(y.values[i] > 5 and y.values[i] <= 48):
         y.values[i] = 1
    elif(y.values[i] > 48 and y.values[i] <= 120):
         y.values[i] = 2

print("Valores",np.unique(y))

#print(y.to_string())



le = preprocessing.LabelEncoder()
y = y.apply(le.fit_transform)



X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size = 0.30,random_state=21)


scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)



param = {'solver': ['adam'],
	'activation' : ['logistic'],
        'max_iter': [900,1000],
        'random_state':[9],
        'hidden_layer_sizes':[10,30,40],
        'alpha': [0.01],
        'learning_rate_init' : [0.001],
        'batch_size':[300],
        #'learning_rate':['constant'],
        'shuffle': [True],
        'momentum':[0.9,0.7],
        #'nesterovs_momentum':[True],
        #'beta_1':[0.1,0.8],
        #'beta_2':[0.999],
        #'verbose':[1],
        'epsilon':[0.00001]}




clf = MLPClassifier()


cv_method = KFold(n_splits=8)

grid_search = GridSearchCV(clf, param_grid=param, n_jobs=-1, cv = cv_method,  error_score='raise')
grid_search.fit(X_train,y_train)

y_pred = grid_search.predict(X_test)
print("Accuracy",accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("confusion_matrix:")
print(cm)
print(classification_report(y_test,y_pred))





#weight = [0.9,1,2,1,1,2,2.1,2.1,2,1,1,1,1,1,1,1,1,1,1,1]














#grid_search = GridSearchCV(MLPRegressor(), param_grid=param, n_jobs=-1, cv = 4,  error_score='raise')
#grid_search.fit(X_train,y_train.values.ravel())
#y_pred = grid_search.predict(X_test)

#print(y_train.values.ravel())
#print(y_pred)
#print(y_test)
print("score: ",grid_search.best_score_)
print("explained_variance_score (1.0)",explained_variance_score(y_test, y_pred))
print("max_error (0.0)",max_error(y_test, y_pred))
print("mean_absolute_error (0.0) ",mean_absolute_error(y_test, y_pred))
print("mean_squared_error (0.0)",mean_squared_error(y_test, y_pred))
#print("mean_squared_log_error (0.0)",mean_squared_log_error(y_test, y_pred))
print("median_absolute_error (0.0)",median_absolute_error(y_test, y_pred))
print("r2_score (1.0)",r2_score(y_test, y_pred))

start = time()
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)

#print(ecolidata)
#mlp = MLPClassifier(hidden_layer_sizes=(tamanioCapasOcultas1,tamanioCapasOcultas2,tamanioCapasOcultas3), max_iter=iterMax, 
#                    alpha=0.0001, solver='sgd', verbose=10,  random_state=21,tol=0.000000001,
#                    learning_rate_init=tasaAprendizajeInicial)


#print(y_train.values.ravel())
#mlp.fit(X_train, y_train.values.ravel())

#predictions = mlp.predict(X_test)
#print(y_test)
#print(predictions)
#print(confusion_matrix(y_test,predictions))  
#print(classification_report(y_test,predictions))

#skplt.metrics.plot_confusion_matrix(y_test, predictions, normalize=True)
#plt.show()

