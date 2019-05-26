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
dataframe = pd.read_csv('Absenteeism_at_work.csv', names=namesColumn, sep=",",header=0)

X = dataframe.iloc[:, 0:20]
y = pd.DataFrame(dataframe["Absenteeism time in hours"])

print(Counter(y.values.ravel()))
print("Valores",np.unique(y))

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


print("Labels: ",Counter(labels))

print("Valores",np.unique(y))
#le = preprocessing.LabelEncoder()
#y = y.apply(le.fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.30,random_state=19)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = MLPClassifier()

#clf.fit(X_train,y_train)
#print("el loss: ",clf.loss_)

param = {'solver': ['sgd'],
		'activation' : ['tanh','relu'],
        'max_iter': [200],
        'random_state':[42],
        'hidden_layer_sizes':[10,15],
        'alpha': [0.01],
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

#print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))

print("Score: ",grid_search.best_score_)

start = time()
print("GridSearchCV duro %.2f segundos para %d configuracion de parametros candidatos."% (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
classes = ["Ausencia Nula","Pocas horas","Muchas horas","Exageradas horas"]


#skplt.metrics.plot_precision_recall_curve(y_test, y_pred)
#plt.plot(clf.loss_)
#snp.labs("number of steps", "loss function", "Loss During GD (Rate=0.001)")


#X_test[:,0]

#Graficas cortas realizadas
#Confusion matrix plot
#skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)

#Cross validation plot
#oz = CVScores(grid_search.best_estimator_, cv=cv_method)
#oz.fit(X_train, y_train)
#oz.poof()

#Learning rate plot
#skplt.estimators.plot_learning_curve(grid_search.best_estimator_, X_train, y_train, cv=cv_method)

########################################################

# No realizadas aun
plt.show()
