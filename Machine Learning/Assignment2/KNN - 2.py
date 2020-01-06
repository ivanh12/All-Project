from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
header = ["Sepal length","Sepal width","Petal length", "Petal width", "Kategori"]
mydataset = pd.read_csv('iris-train.csv', sep=',', names=header)
mytestset = pd.read_csv('iris-test.csv', sep=',', names=header)

mydataset.loc[mydataset['Kategori'] == "Iris-setosa", 'Kategori'] = 0
mydataset.loc[mydataset['Kategori'] == "Iris-versicolor", 'Kategori'] = 1
mydataset.loc[mydataset['Kategori'] == "Iris-virginica", 'Kategori'] = 2

mytestset.loc[mytestset['Kategori'] == "Iris-setosa", 'Kategori'] = 0
mytestset.loc[mytestset['Kategori'] == "Iris-versicolor", 'Kategori'] = 1
mytestset.loc[mytestset['Kategori'] == "Iris-virginica", 'Kategori'] = 2

X = mydataset.iloc[:,0:4].values
y = mydataset.iloc[:,- 1].values

X_test = mytestset.iloc[:,0:4].values
y_test = mytestset.iloc[:,- 1].values

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=1)
knn2 = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn3 = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

scores = cross_validate(knn, X, y, cv=10, return_train_score=False)
scores2 = cross_validate(knn2, X, y, cv=10, return_train_score=False)
scores3 = cross_validate(knn2, X, y, cv=10, return_train_score=False)

#y_pred = cross_val_predict(knn, X, y, cv=10)
accuracy_train = 0
accuracy_train2 = 0
accuracy_train3 = 0
for i in range(10):
    accuracy_train = accuracy_train + scores['test_score'][i]
    accuracy_train2 = accuracy_train2 + scores2['test_score'][i]  
    accuracy_train3 = accuracy_train3 + scores3['test_score'][i]   
    
accuracy_train = accuracy_train / 10 * 100
accuracy_train2 = accuracy_train2 / 10 * 100
accuracy_train3 = accuracy_train3 / 10 * 100
print("Accuracy Training (Manhattan) : ", '%.2f' % accuracy_train)
print("Accuracy Training (Euclidean) : ", '%.2f' % accuracy_train2)
print("Accuracy Training (Minkowski) : ", '%.2f' % accuracy_train3)
print("")

"""Akurasi test data"""
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=1)
classifier2 = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
classifier3 = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

classifier.fit(X, y)
classifier2.fit(X, y)
classifier3.fit(X, y)

accuracy_test = classifier.score(X_test, y_test)
accuracy_test = accuracy_test * 100

accuracy_test2 = classifier2.score(X_test, y_test)
accuracy_test2 = accuracy_test2 * 100

accuracy_test3 = classifier3.score(X_test, y_test)
accuracy_test3 = accuracy_test3 * 100
print("Accuracy Test (Manhattan) : ", '%.2f' % accuracy_test)
print("Accuracy Test (Euclidean) : ", '%.2f' % accuracy_test2)
print("Accuracy Test (Minkowski) : ", '%.2f' % accuracy_test3)

#y_pred = classifier.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)*100
#print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')



