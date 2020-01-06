# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score


header = ["Age","Sex","Chest Pain", "Blood Pressure", "Cholestrol Level", "Blood Sugar", "ECG Results", "Maximum Heart Rate", 
          "Exercise-Induced Angina", "Exercise-Induced ST", "ST Segment Peak Slope", "Fluoroscopy Result", "Thalassemia", "Target"]

mydataset = pd.read_csv('heartdisease-train.csv', names=header)
testdata = pd.read_csv('heartdisease-test.csv', names=header)

X = mydataset.iloc[:,0:14].values
Y = mydataset.iloc[:,- 1].values
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,random_state = 10)

print("\nTrain Data")
BernNB = BernoulliNB(binarize = True)
BernNB.fit(X_train, Y_train)
y_expect = Y_test
y_pred = BernNB.predict(X_test)
BernTrainAcc = accuracy_score(y_expect, y_pred)
print("\nAccuracy Bernoulli Train Data : ", BernTrainAcc)
print ("Cross Validation Bernoulli Train Data : ", cross_val_score(BernNB,X,Y,cv=10))

MultiNB = MultinomialNB()
MultiNB.fit(X_train, Y_train)
y_pred = MultiNB.predict(X_test)
MultiTrainAcc = accuracy_score(y_expect,y_pred)
print("\nAccuracy Multinomial Train Data : ",MultiTrainAcc)
print ("Cross Validation Bernoulli Train Data : ", cross_val_score(MultiNB,X,Y,cv=10))

GausNB = GaussianNB()
GausNB.fit(X_train , Y_train)
y_pred = GausNB.predict(X_test)
GausTrainAcc = accuracy_score(y_expect,y_pred)
print ("\nAccuracy Gaussian Train Data : ", GausTrainAcc)
print ("Cross Validation Bernoulli Train Data : ", cross_val_score(GausNB,X,Y,cv=10))

count_row = mydataset.shape[0]
count_col = mydataset.shape[1]

print("\nTest Data")

X_t = testdata.iloc[:,0:14].values
Y_t = testdata.iloc[:,- 1].values
X_train , X_test , Y_train , Y_test = train_test_split(X_t,Y_t,random_state = 10)

BernNB = BernoulliNB(binarize = True)
BernNB.fit(X_train, Y_train)
y_expect = Y_test
y_pred = BernNB.predict(X_test)
BernTestAcc = accuracy_score(y_expect,y_pred)
print("\nAccuracy Bernoulli Test Data : ", BernTestAcc)
print ("Cross Validation Gaussian Test Data : ", cross_val_score(BernNB,X_t,Y_t,cv=10))

MultiNB = MultinomialNB()
MultiNB.fit(X_train, Y_train)
y_pred = MultiNB.predict(X_test)
MultiTestAcc = accuracy_score(y_expect,y_pred)
print("\nAccuracy Multinomial Test Data : ",MultiTestAcc)
print ("Cross Validation Gaussian Test Data : ", cross_val_score(MultiNB,X_t,Y_t,cv=10))

GausNB = GaussianNB()
GausNB.fit(X_train , Y_train)
y_pred = GausNB.predict(X_test)
GausTestAcc = accuracy_score(y_expect,y_pred)
print ("\nAccuracy Gaussian Test Data : ", GausTestAcc)
print ("Cross Validation Gaussian Test Data : ", cross_val_score(GausNB,X_t,Y_t,cv=10))
