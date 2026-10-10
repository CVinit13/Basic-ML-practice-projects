import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

mnist = fetch_openml('mnist_784', version=1)
X, Y = mnist['data'], mnist['target']

# Print information about the dataset
print(f"Number of samples in the dataset: {len(X)}")
print(f"Shape of the feature matrix (X): {X.shape}")
print(f"Shape of the target vector (Y): {Y.shape}")

# Check if index 3601 is valid
if len(X) > 3601:
    some_digit = X.iloc[3601].to_numpy()
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=plt.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

    # Rest of your code...
    X_train, X_test, Y_train, Y_test = X.iloc[0:60000], X.iloc[60000:70000], Y.iloc[:60000], Y.iloc[60000:70000]
    shuffle_index = np.random.permutation(len(X_train))
    X_train, Y_train = X_train.iloc[shuffle_index], Y_train.iloc[shuffle_index]
    Y_train = Y_train.astype(np.int8)
    Y_test = Y_test.astype(np.int8)
    Y_train_2 = (Y_train == 2)
    Y_test_2 = (Y_test == 2)

    clf = LogisticRegression(tol=0.1)
    clf.fit(X_train, Y_train_2)
    a = clf.predict([some_digit])  # Use [some_digit] to make it a 2D array
    b = cross_val_score(clf, X_train, Y_train_2, cv=3, scoring="accuracy")

    print(a)
    print(b.mean())
else:
    print("Index 3601 is not valid for the given dataset.")

