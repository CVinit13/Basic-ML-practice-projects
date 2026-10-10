# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
mnist = fetch_openml('mnist_784',version=1)
X, Y = mnist['data'], mnist['target']
some_digit = X[3601]
some_digit_image = some_digit.mpl.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
X_train, X_test, Y_train, Y_test = X[0:60000],X[60000:70000],Y[:60000],Y[60000:70000]
shuffle_index = np.random.permutation()
X_train, Y_train = X_train[shuffle_index], Y_train[shuffle_index]
Y_train = Y_train.astype(np.int8)
Y_test = Y_test.astype(np.int8)
Y_train_2 = (Y_train==2)
Y_test_2 = (Y_test==2)
clf = LogisticRegression(tol=0.1)
clf.fit(X_train, Y_train_2)
a=clf.predict(some_digit)
b = cross_val_score(clf,X_train,Y_train_2,cv = 3, scoring="acuracy")
print(a)
print(b.mean)