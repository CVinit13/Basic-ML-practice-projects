# Loading Required Moduels
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Loading Dataset
iris = datasets.load_iris()

# Printing Discription and features
features = iris.data
labels = iris.target
print(features[0],labels[0])
print(iris.DESCR)

# Training the classifiers
clf = KNeighborsClassifier()
clf.fit(features,labels)

preds = clf.predict([[6.1,3.3,7.1,3.3]])
print(preds)
