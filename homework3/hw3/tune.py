import numpy as np
from dataset import get_data,get_HOG,standardize

######################## Get train/test dataset ########################
X_train,X_test,Y_train,Y_test = get_data('dataset')
########################## Get HoG featues #############################
H_train,H_test = get_HOG(X_train), get_HOG(X_test)
######################## standardize the HoG features ####################
H_train,H_test = standardize(H_train), standardize(H_test)

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

### Tune rbf hyper-parameters
print(f"Default gamma of SVC: {1 / (H_train.shape[1] * H_train.var())}")
param_grid = {'gamma': [0.002, 0.003, 0.004, 0.005, 0.006, 0.007]}
svm = SVC(kernel = 'rbf')
grid_search = GridSearchCV(svm, param_grid, scoring='accuracy', cv=5)
grid_search.fit(H_train, Y_train)
print('Best parameters:', grid_search.best_params_)
print('Best score:', grid_search.best_score_)

### Tune poly hyper-parameters
param_grid = {'coef0': [0.125, 0.25, 0.375], 'degree': [3, 4, 5, 6]}
svm = SVC(kernel='poly')
grid_search = GridSearchCV(svm, param_grid, scoring='accuracy', cv=5)
grid_search.fit(H_train, Y_train)
print('Best parameters:', grid_search.best_params_)
print('Best score:', grid_search.best_score_)