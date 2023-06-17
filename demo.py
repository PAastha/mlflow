import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

lr = LogisticRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)

accuracy = accuracy_score(y_test,pred)

print(accuracy)
