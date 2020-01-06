# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 09:59:56 2018

@author: asus
"""
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd

min_max_scaler = preprocessing.MinMaxScaler()

mydataset = pd.read_csv('breast-cancer-data.csv')
mydataset.loc[mydataset['diagnosis'] == "B", 'diagnosis'] = 0
mydataset.loc[mydataset['diagnosis'] == "M", 'diagnosis'] = 1

X = mydataset.iloc[:,2:32].values
y = mydataset.iloc[:,1].values



X_minmax = min_max_scaler.fit_transform(X)

"""Akurasi train data"""
#print("Before Scaling")
Y_man = []
Y_euc = []
Y_min = []

for K in range (3, 10, 2):
    classifier = KNeighborsClassifier(n_neighbors=K, metric='minkowski', p=1)
    classifier2 = KNeighborsClassifier(n_neighbors=K, metric='euclidean')
    classifier3 = KNeighborsClassifier(n_neighbors=K, metric='minkowski', p=3)
    
    classifier.fit(X, y)
    classifier2.fit(X, y)
    classifier3.fit(X, y)
    
    accuracy_test = classifier.score(X, y)
    accuracy_test = accuracy_test * 100
    Y_man.append(accuracy_test)
    
    accuracy_test2 = classifier2.score(X, y)
    accuracy_test2 = accuracy_test2 * 100
    Y_euc.append(accuracy_test2)
    
    accuracy_test3 = classifier3.score(X, y)
    accuracy_test3 = accuracy_test3 * 100
    Y_min.append(accuracy_test3)
    
    """print("K = %.0f" % K)
    print("Accuracy Train (Manhattan)        : ", '%.2f' % accuracy_test)
    print("Accuracy Train (Euclidean)        : ", '%.2f' % accuracy_test2)
    print("Accuracy Train (Minkowski, p = 3) : ", '%.2f' % accuracy_test3)
    print("")"""
    
    #y_pred = classifier.predict(X)
    #accuracy = accuracy_score(y, y_pred)*100
    #print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
    
    
Y_man_cv = []
Y_euc_cv = []
Y_min_cv = []  
for K in range(3, 10, 2): 
    knn = KNeighborsClassifier(n_neighbors=K, metric='minkowski', p=1)
    knn2 = KNeighborsClassifier(n_neighbors=K, metric='euclidean')
    knn3 = KNeighborsClassifier(n_neighbors=K, metric='minkowski', p=3)
    
    scores = cross_validate(knn, X, y, cv=10, return_train_score=False)
    scores2 = cross_validate(knn2, X, y, cv=10, return_train_score=False)
    scores3 = cross_validate(knn3, X, y, cv=10, return_train_score=False)
    
    accuracy_train = 0
    accuracy_train2 = 0
    accuracy_train3 = 0
    for i in range(10):
        accuracy_train = accuracy_train + scores['test_score'][i]
        accuracy_train2 = accuracy_train2 + scores2['test_score'][i]
        accuracy_train3 = accuracy_train3 + scores3['test_score'][i]    
    accuracy_train = accuracy_train / 10 * 100
    Y_man_cv.append(accuracy_train)
    
    accuracy_train2 = accuracy_train2 / 10 * 100
    Y_euc_cv.append(accuracy_train2)
    
    accuracy_train3 = accuracy_train3 / 10 * 100
    Y_min_cv.append(accuracy_train3)
    
    """print("K = %.0f" % K)
    print("Cross Validation Accuracy (Manhattan)        : ", '%.2f' % accuracy_train)
    print("Cross Valudation Accuracy (Euclidean)        : ", '%.2f' % accuracy_train2)
    print("Cross Validation Accuracy (Minkowski, p = 3) : ", '%.2f' % accuracy_train3)
    print("")"""
    
print("")

"""Setelah Scaling"""
#print("After Scaling")
Y_man_minmax = []
Y_euc_minmax = []
Y_min_minmax = []

for K in range (3, 10, 2):
    classifier = KNeighborsClassifier(n_neighbors=K, metric='minkowski', p=1)
    classifier2 = KNeighborsClassifier(n_neighbors=K, metric='euclidean')
    classifier3 = KNeighborsClassifier(n_neighbors=K, metric='minkowski', p=3)
    
    classifier.fit(X_minmax, y)
    classifier2.fit(X_minmax, y)
    classifier3.fit(X_minmax, y)
    
    accuracy_test = classifier.score(X_minmax, y)
    accuracy_test = accuracy_test * 100
    Y_man_minmax.append(accuracy_test)
    
    accuracy_test2 = classifier2.score(X_minmax, y)
    accuracy_test2 = accuracy_test2 * 100
    Y_euc_minmax.append(accuracy_test2)
    
    accuracy_test3 = classifier3.score(X_minmax, y)
    accuracy_test3 = accuracy_test3 * 100
    Y_min_minmax.append(accuracy_test3)
    
    """print("Accuracy Train (Manhattan)        : ", '%.2f' % accuracy_test)
    print("Accuracy Train (Euclidean)        : ", '%.2f' % accuracy_test2)
    print("Accuracy Train (Minkowski, p = 3) : ", '%.2f' % accuracy_test3)
    print("")"""

#y_pred = classifier.predict(X)
#accuracy = accuracy_score(y, y_pred)*100
#print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')

Y_man_cv_minmax = []
Y_euc_cv_minmax = []
Y_min_cv_minmax = []

for K in range(3, 10, 2):
    knn = KNeighborsClassifier(n_neighbors=K, metric='minkowski', p=1)
    knn2 = KNeighborsClassifier(n_neighbors=K, metric='euclidean')
    knn3 = KNeighborsClassifier(n_neighbors=K, metric='minkowski', p=3)
    
    scores = cross_validate(knn, X_minmax, y, cv=10, return_train_score=False)
    scores2 = cross_validate(knn2, X_minmax, y, cv=10, return_train_score=False)
    scores3 = cross_validate(knn3, X_minmax, y, cv=10, return_train_score=False)
    
    accuracy_train = 0
    accuracy_train2 = 0
    accuracy_train3 = 0
    for i in range(10):
        accuracy_train = accuracy_train + scores['test_score'][i]
        accuracy_train2 = accuracy_train2 + scores2['test_score'][i]
        accuracy_train3 = accuracy_train3 + scores3['test_score'][i]    
        
    accuracy_train = accuracy_train / 10 * 100
    Y_man_cv_minmax.append(accuracy_train)
    
    accuracy_train2 = accuracy_train2 / 10 * 100
    Y_euc_cv_minmax.append(accuracy_train2)
    
    accuracy_train3 = accuracy_train3 / 10 * 100
    Y_min_cv_minmax.append(accuracy_train3)
    
    """print("Cross Validation Accuracy (Manhattan)        : ", '%.2f' % accuracy_train)
    print("Cross Valudation Accuracy (Euclidean)        : ", '%.2f' % accuracy_train2)
    print("Cross Validation Accuracy (Minkowski, p = 3) : ", '%.2f' % accuracy_train3)"""


i = 0
print("                                     Before Scaling   After Scaling")
for K in range (3, 10, 2):
    print ("K = %.0f" % K)
    print("Train Accuracy (Manhattan)        : ", '%.2f %16.2f' % (Y_man[i], Y_man_minmax[i]))
    print("Train Accuracy (Euclidean)        : ", '%.2f %16.2f' % (Y_euc[i], Y_euc_minmax[i]))
    print("Train Accuracy (Minkowski, p = 3) : ", '%.2f %16.2f' % (Y_min[i], Y_min_minmax[i]))
    i = i + 1
    print("")

i = 0
print("                                                Before Scaling   After Scaling")
for K in range (3, 10, 2):
    print ("K = %.0f" % K)
    print("Cross Validation Accuracy (Manhattan)        : ", '%.2f %16.2f' % (Y_man_cv[i], Y_man_cv_minmax[i]))
    print("Cross Validation Accuracy (Euclidean)        : ", '%.2f %16.2f' % (Y_euc_cv[i], Y_euc_cv_minmax[i]))
    print("Cross Validation Accuracy (Minkowski, p = 3) : ", '%.2f %16.2f' % (Y_min_cv[i], Y_min_cv_minmax[i]))
    i = i + 1
    print("")