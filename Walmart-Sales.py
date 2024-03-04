#!/usr/bin/env python
# coding: utf-8

# # Walmart Sales Prediction Project

# Database Description:-
# 
# Organisation Name: Walmart
# 
# Kaggle Data Source : https://www.kaggle.com/datasets/mikhail1681/walmart-sales
# 
# Algorithms Applied:
# 1) Linear Regression
# 
# 2) Random Forest Regressor
# 
# Time Period (Year): 2010-2012
# 
# Objective:-
# 
# Weekly sales predication model of Walmart store.    

# # Nutshell:

# In[2]:


# Important Libraries:
import cv2 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[6]:


Model_In_Nutshell = cv2.imread('Screenshot (227).png')
plt.figure(figsize=(50,38))
plt.imshow(Model_In_Nutshell)
plt.show()


# # Sales Database Acquisition:

# In[49]:


Walmart = pd.read_csv('Walmart_sales.csv')
Walmart


# # Data-Cleaning:

# In[50]:


#Extracting information about the data type of each column.
Walmart.info()


# In[51]:


# Checking for null values.
Walmart.isnull()
##Therefore, there are no null values in the database.


# In[52]:


#Displaying all the rows of the database.
pd.pandas.set_option('display.max_rows',None)


# In[53]:


Walmart


# In[54]:


# Changing the decimal places of figures under Fuel_Price, CPI and Unemployment rate upto 2 decimal places.
def Decimal_Places(column):
    return round(column,2)


# In[55]:


Walmart['Fuel_Price'] = Walmart['Fuel_Price'].apply(Decimal_Places)
Walmart['CPI'] = Walmart['CPI'].apply(Decimal_Places)
Walmart['Unemployment rate (%)'] = Walmart['Unemployment rate (%)'].apply(Decimal_Places)
Walmart


# In[56]:


# Rounding decimals upto 2 places.
Walmart[['Weekly_Sales','Holiday_Flag','Temperature','CPI','Unemployment rate (%)','Fuel_Price']] = Walmart[['Weekly_Sales','Holiday_Flag','Temperature','CPI','Unemployment rate (%)','Fuel_Price']].apply(Decimal_Places)
Walmart


# Reggression Analysis :
# We will be applying linear regression algorithm in order to predict the weekly sales of Walmart stores.
# For this, purpose we will be linear relations between the weekly sales as our independent variables and other variables which will be chosen based on correlation heatmap.  

# In[57]:


# Developing correlation heatmap to analyse the relationship between weekly sales and other variables.
Correlation = Walmart.corr()
Correlation
# Therefore, we will be eradicating Holiday_flag from our regression equation since it does not have any relationship with the storw's weekly sale.


# In[58]:


# Assinging dependent variable:
y = Walmart['Weekly_Sales']


# In[59]:


# It should be noted that we cannot include Holiday_flag under our linear regression model since it consists of binary values (0 and 1). Therefore, Logistic regression shall be suitable for this modeling process.  
# Assinging independent variable:

X = Walmart[['Temperature','Fuel_Price','CPI','Unemployment rate (%)']]   


# We will be performing train test split of the database. For this purpose, we will be assinging 30% of the database to test set and 70% for training data set.
# 

# # Model Training and Testing:

# In[60]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


# In[61]:


X_train # Training dataset.


# In[62]:


X_test      # Testing dataset


# In[63]:


y_train   # Training y dataset


# In[64]:


y_test # Testing dataset


# In[65]:


### We will be fitting our regression model to the training dataset. For which, we will have to import our linear regression model first.
from sklearn.linear_model import LinearRegression


# In[66]:


Model = LinearRegression()
Model.fit(X_train,y_train)


# In[67]:


######## Extracting independent variable cofficients from the equation.
print(Model.coef_)


# In[68]:


######## Extracting variable intercept from the equation.
print(Model.intercept_)


# In[69]:


############Dataframing the cofficients.
pd.DataFrame(Model.coef_,X.columns,columns=['Coefficients'])


# In[70]:


####### Creating predictions of dependent variables in Test dataset.
y_Predicted_values = Model.predict(X_test)
y_Predicted_values


# In[71]:


y_Predicted_values = np.round(y_Predicted_values,2)
y_Predicted_values


# In[72]:


####### Creating dataframe of y_predicted that is predicated sales of each stores.
pd.DataFrame(y_Predicted_values,columns=['Predicted Sales'])            


# Model Evaluation: For performing model evalutation, under which we will be using 3 metrics which are:-
# 1) Coffiecient of determination (R^2): Under this method there is a scale between 0 and 1, we test our pramaters under test dataset in order to evaluate our        model that has been applied on training dataset. This method is Coffiecient of Demtermination. If the value is 1, then ot        means that our model is perfect otherwase it is poor.
# 
# 2) Mean Absolute Error: 
# 
# 3) Mean Squared Error:

# In[87]:


#1) Cofficient of determination (R^2): Round cofficient of determination upto 3 decimal places.
from sklearn.metrics import r2_score  #Using cofficient of determination for evaluate the model.
COD = np.round(r2_score(y_test,y_Predicted_values),3)
print(f'The cofficient of determination is {COD}')


# In[88]:


from sklearn import metrics


# In[89]:


#2) Mean Absolute Error: Rounding up the mean absolute error upto 3 decimal places.
MAE = np.round(metrics.mean_absolute_error(y_test,y_Predicted_values),3)
print(f'Mean absolute error is {MAE}')


# In[90]:


#3) Mean Squared Error: Rounding up the mean squared error upto 3 decimal places.
MSE = np.round(metrics.mean_squared_error(y_test,y_Predicted_values),3)
print(f'Mean squared error is {MSE}')


# In[151]:


####Distribution plot: Weeky Sales V/S Predicted Weekly Sales
plt.figure(figsize=(20,10))
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
sns.distplot(y_Predicted_values,bins=5,label='Predicted sales',color='g',hist_kws=dict(linewidth=50)) # Predicted Sales
sns.distplot(y_test,bins=5,label='Actual sales',color='r',hist_kws=dict(linewidth=50))  # Actual Sales
plt.xlabel('Weekly Sales',fontsize=25)
plt.ylabel('Density',fontsize=25)
plt.grid(color='black', linestyle='-', linewidth=1)
plt.title('Actual Sales V/S Predicted Sales',fontsize=25)
plt.legend(fontsize=25)
plt.show()


# Problem:
# 1) The cofficient of determination (R^2) is 0.014.
# 
# 2) Therefore, the linear regression model suggests that independent varaiables account for 1.4% of variability in the weekly sales Walmart. 
# 
# 3) Thus, the sales predictions made by the model are not fully accurate and requires further adjustments in the model. 

# Solution:
# In order to make our sales predictions more accurate and reduce our model errors, we will be applying a robust algorithm called Random Forest Regressoion on the dataset.  
#      

# # Model Adjustment:

# Modified-Model:-
# 
# Algorithm: Random Forest Regressor
# 
#  

# In[152]:


# Importing random forest regressor. Because model is unfit, there we will be applyignt random forest regressor.
from sklearn.ensemble import RandomForestRegressor


# In[153]:


New_Model = RandomForestRegressor()
New_Model.fit(X_train,y_train)


# In[165]:


# Predicted weekly sales under modified model.
Y_predicted_Values = New_Model.predict(X_test)                 
Y_predicted_Values


# Model Evaluation:  

# In[171]:


# Cofficient of determination:
C_O_D = np.round(r2_score(y_test,Y_predicted_Values),3)
print(f'The cofficient of determination is {C_O_D}')


# In[173]:


# Mean squared Error:
M_S_E = np.round(metrics.mean_squared_error(y_test,Y_predicted_Values),3)
print(f'Mean squared error of modified model: {M_S_E}')


# In[174]:


# Mean Absolute Error:
M_A_E = np.round(metrics.mean_absolute_error(y_test,Y_predicted_Values),3)
print(f'Mean absolute error of modified model: {M_A_E}' )


# In[169]:


# Distribution plot of modified model: 
plt.figure(figsize=(20,10))
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
sns.distplot(Y_predicted_Values,bins=5,label='Predicted sales',color='g',hist_kws=dict(linewidth=50)) # Predicted Sales
sns.distplot(y_test,bins=5,label='Actual sales',color='r',hist_kws=dict(linewidth=50))  # Actual Sales
plt.xlabel('Weekly Sales',fontsize=25)
plt.ylabel('Density',fontsize=25)
plt.grid(color='black', linestyle='-', linewidth=1)
plt.title('Actual Sales V/S Predicted Sales',fontsize=25)
plt.legend(fontsize=25)
plt.show()


# Insight:
#     
# 1) Cofficient of determination = 0.156
#    The independent variables account for 15.6% of the varuablity in weekly sales.
#     
# 2) Mean Squared Error: 261838679454.481
#     
# 3) Mean Absolute Error: 369518.902

# # Conclusion:
# 
# The impact after the application of new algorithm to the model can be explained as follows:-

# 1) Cofficient of determination:

# In[197]:


Model_One = 0.014
Modified_Model = 0.156 
Increase = Modified_Model-Model_One
print(f'There has been an increase in the cofficient of determination (R^2) by :{Increase}') 


# 2) Mean Squared Error:

# In[193]:


Model_One =  305720188756.577
Modified_Model = 261838679454.481
Decrease = np.round(((261838679454.481-305720188756.577)/ 305720188756.577)*100,2)
print(f'There has been a precentage decrease in the dispersion between predicted and actual sales by :{Decrease} %') 


# 3) Mean Absolute Error:

# In[198]:


Model_One = 467439.67
Modified_Model = 369518.902
Decrease = np.round(((Modified_Model-Model_One)/Model_One)*100,2)
print(f'There has been a precentage decrease in the absolute difference between predicted and actual sales by :{Decrease} %') 


# # Deployment:
# 
# Therefore, the model is ready to be deployed for sales prediction process.
# 
# 
