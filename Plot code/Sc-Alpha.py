import pandas as pd
from time import time
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import precision_score,explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score, classification_report,confusion_matrix, accuracy_score
import scikitplot as skplt
import matplotlib.pyplot as plt
from collections import Counter
from yellowbrick.model_selection import CVScores

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Modelo con rank: {0}".format(i))
            print("Precision de validacio promedio: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parametros: {0}".format(results['params'][candidate]))
            print("")

namesColumn = ["ID", "Reason for absence","Month of absence", "Day of the week", "Seasons", "Transportation expense", "Distance from Residence to Work", "Service time", "Age", "Work load Average/day ", "Hit target", "Disciplinary failure", "Education", "Son", "Social drinker", "Social smoker", "Pet", "Weight","Height","Body mass index","Absenteeism time in hours"]
dataframe = pd.read_csv('./../Absenteeism_at_work.csv', names=namesColumn, sep=",",header=0)

X = dataframe.iloc[:, 0:20]
y = pd.DataFrame(dataframe["Absenteeism time in hours"])

labels = ["" for x in range(y.size)]

for idx,item in enumerate(y.values, start=0):
    if(item == 0):
        labels[idx] = "Ausencia Nula"
    elif(item > 0 and item <= 40):
        labels[idx] = "Pocas horas"
    elif(item > 40 and item <= 80):
        labels[idx] = "Muchas horas"
    elif(item > 80 and item <= 120):
        labels[idx] = "Exageradas horas"


X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.30,random_state=19)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = MLPClassifier()

alphas = [0.0001,0.001,0.01,0.1,0.9]
activation = ['tanh','relu','logistic']
param = {'solver': ['sgd'],
		'activation' : activation,
        'max_iter': [200],
        'random_state':[42],
        'hidden_layer_sizes':[15],
        'alpha': alphas,
        'learning_rate_init' : [0.001],
        'batch_size':[200],
        'learning_rate':['constant'],
        'shuffle': [True],
        'momentum':[0.9],
        #'verbose':[1],
		#'power_t':[0.1],
		#'warm_start':[True],
		#'early_stopping':[False],
}

cv_method = KFold(n_splits=6)

grid_search = GridSearchCV(clf, param_grid=param, n_jobs=-1, cv = cv_method, scoring='accuracy', return_train_score=True)
grid_search.fit(X_train,y_train)
y_pred = grid_search.predict(X_test)

print("Score: ",grid_search.best_score_)

start = time()
print("GridSearchCV duro %.2f segundos para %d configuracion de parametros candidatos."% (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
classes = ["Ausencia Nula","Pocas horas","Muchas horas","Exageradas horas"]

print(grid_search.cv_results_['mean_test_score'])

def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    _, ax = plt.subplots(1,1)

    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Activation Function - Alpha Ratio", fontsize=20)
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')

plot_grid_search(grid_search.cv_results_, alphas, activation, 'alpha rate', 'activation function' )
plt.show()
