import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv(r"C:\Users\DELL\Downloads\mobile_dataset_600.csv")
print(data)
print(data.columns)
print(data.isnull().sum())
data.duplicated().isnull().sum()
x=data[['Model_Year','RAM (GB)','Storage (GB)','Battery (mAh)','Primary Camera (MP)']]
y=data['Price (USD)']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
print(x_train)      
print(x_test)
print(y_train)
print(y_test)
from sklearn.linear_model import LinearRegression
model2=LinearRegression()
model2.fit(x_train,y_train)
ypred=model2.predict(x_test)
print(ypred)
print(y_test)
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
r2=r2_score(y_test,ypred)
mse=mean_squared_error(y_test,ypred)
mae=mean_absolute_error(y_test,ypred)
print(r2)
print(mse)
print(mae)
ypred=model2.predict([[2020,8,128,5449,105]])
print(ypred)
import pickle as pkl
pkl.dump(model2,open(r'model2_mob.pkl','wb'))
