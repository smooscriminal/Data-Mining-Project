from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, SVC
import cv2
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def gridsearch(model,param_grid,scoring='f1_macro'):
    clf = GridSearchCV(model, param_grid, cv=5,
                       scoring=scoring,return_train_score=True)
    clf.fit(X,y)
    print("Best parameters set found on development set:")
    print('%r\nTrain: %.3f Test: %.3f' % (clf.best_params_,clf.cv_results_['mean_train_score'][clf.best_index_],clf.cv_results_['mean_test_score'][clf.best_index_]))
    return clf.best_estimator_

def classifying_report(clf, X_test):
    y_pred = clf.predict(X_test)
    print
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('\nAccuracy: %.4f' % accuracy_score(y_test,y_pred))
    print
    print("Classification Report:")
    print(classification_report(y_test,y_pred,labels=sorted(set(y)),digits=4))

filename = ''

data = pd.read_csv(filename)



