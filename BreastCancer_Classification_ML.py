from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

bCancer = load_breast_cancer()
#print(bCancer['data'].shape)
#print(bCancer['data'])
#print(bCancer)
X_egitim, X_test, y_egitim, y_test = train_test_split(bCancer['data'], bCancer['target'], random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_egitim, y_egitim)

print(knn.score(X_test,y_test))


