import numpy as np 
import pandas as pd
import mlflow

from sklearn.neighbors import KNeighborsClassifier
from sklearn. model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

cancer=load_breast_cancer()


cancer['feature_names']
df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

scaler = StandardScaler()

scaler.fit(df_feat)


scaled_features = scaler.transform(df_feat)
df_target = np.ravel(cancer['target'])


X_train, X_test, y_train, y_test = train_test_split(scaled_features, df_target, test_size=0.30, random_state=105)


knn = KNeighborsClassifier(n_neighbors=3)


knn.fit(X_train, y_train)
y_predicted = knn.predict(X_test)
print(accuracy_score(y_test, y_predicted))