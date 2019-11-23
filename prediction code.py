import numpy as np
import pandas as pd
import numpy as np
import math
import sklearn
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing
import csv
import matplotlib.pyplot as plt 
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data=pd.read_csv("dataset1.csv")
print("Data heads:")
print(data.head())
print("Null values in the dataset before preprocessing:")
print(data.isnull().sum())
print("Filling null values with mean of that particular column")
data=data.fillna(np.mean(data))
print("Mean of data:")
print(np.mean(data))
print("Null values in the dataset after preprocessing:")
print(data.isnull().sum())
print("\n\nShape: ",data.shape)

 
print("Info:")
print(data.info())
 
print("Group by:")
data.groupby('field1').size()


print("Group by:")
data.groupby('field2').size()


print("Co-Variance =",data.cov())
print("Co-Relation =",data.corr())
 
print("Scatter plot of field1 and field2 attributes")
plt.scatter(data.field1,data.field2)
plt.xlabel("Temperature")
plt.ylabel("humidity")
Scatter plot of field1 and field2 attributes
Text(0,0.5,'humidity')

max1= data.field1.max() 
min1= data.field1.min()
max2=data.field2.max()
min2= data.field2.min()
print("maximum temperature of the day is:" ,max1)
print("minimum temperature of the day is:" ,min1)
print("maximum humidity of the day is:" ,max2)
print("minimum humidity of the day is:" ,min2)


plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(data['field1'])
plt.title("temperature range")
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(data['field2'])
plt.title("humidity range")

Text(0.5,1,'humidity range')


x= data['field2'].values.reshape(-1,1)
y = data['field1'].values.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(x_train, y_train) #training the algorithm
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)
[27.12226153]
[[0.01938947]]
y_pred=regressor.predict(x_test)

df1=pd.DataFrame({'Actual1': y_test.flatten(), 'Predicted1': y_pred.flatten()})
df1

df3=df1.head(25)
df3.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title("temperature predictions")
plt.show()


plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_pred, color='red', linewidth=3)
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

x1= data['field1'].values.reshape(-1,1)
y1= data['field2'].values.reshape(-1,1)
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(x1_train, y1_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)

y1_pred = regressor.predict(x1_test)

df2 = pd.DataFrame({'Actual2': y1_test.flatten(), 'Predicted2': y1_pred.flatten()})
df2

df4=df2.head(25)
df4.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title("humidity predictions")
plt.show()

plt.scatter(x1_test, y1_test,  color='black')
plt.plot(x1_test, y1_pred, color='red', linewidth=3)
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

max3=y_pred.max() 
min3=y_pred.min()
max4=y1_pred.max()
min4=y1_pred.min()
print("maximum predicted temperature is:" ,max3)
print("minimum predicted temperature is:" ,min3)
print("maximum predicted humidity is:" ,max4)
print("minimum predicted humidity is:" ,min4)

