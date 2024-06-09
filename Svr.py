import pandas as pd
import numpy as np

df=pd.read_csv(r"D:\coding journey\aiml\python\udemy\Machine Learning A-Z (Codes and Datasets)\Part 2 - Regression\Section 7 - Support Vector Regression (SVR)\Python\Position_Salaries.csv")
x=df.iloc[:, 1:-1].values
y=df.iloc[:, -1].values
y=y.reshape(len(y),1)
print(y)
from sklearn.preprocessing import StandardScaler
st_1=StandardScaler()
y=st_1.fit_transform(y)
st_2=StandardScaler()
x=st_2.fit_transform(x)
print(x,"\n")
print(y)

from sklearn.svm import SVR
sv=SVR(kernel="rbf")
sv.fit(x,y)

y_predict=st_1.inverse_transform(sv.predict(st_2.transform([[6.5]])).reshape(-1,1))
print(y_predict)


