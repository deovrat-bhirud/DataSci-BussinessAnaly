import pandas as pd  
import numpy as np    
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# importing the dataset
data_set = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
data_set.head()  

# Plotting Dataset
data_set.plot(x='Hours', y='Scores', style='o')    
plt.title('Hours vs Percentage')    
plt.xlabel('The Hours Studied')    
plt.ylabel('The Percentage Score')    
plt.show() 

# data preprocessing
x = data_set.iloc[:, :-1].values   #independent variable array
y = data_set.iloc[:, 1].values     #dependent variable vector 

#Train the data  
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0)

# fitting the regression model
regressor = LinearRegression()    
regressor.fit(x_train, y_train)
line = regressor.coef_*x+regressor.intercept_  


plt.scatter(x, y)  
plt.plot(x, line);  
plt.show() 

# predicting the test set results
y_pred = regressor.predict(x_test) 
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})    
df 


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

hours = [[9.25]]  
own_pred = regressor.predict(hours)  
print("Number of hours = {}".format(hours))  
print("Prediction Score = {}".format(own_pred[0])) 
