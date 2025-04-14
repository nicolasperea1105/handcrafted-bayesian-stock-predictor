# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 23:39:01 2025

@author: nicol
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import joblib

online = pd.read_pickle("my_file_path")
InStore = pd.read_pickle("my_file_path")

## Functions needed

def regression_metrics(actual, predicted):
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    mae = np.mean(np.abs(predicted - actual))
    r2 = 1 - np.sum((actual - predicted)**2) / np.sum((actual - np.mean(actual))**2)
    return rmse, mae, r2


## General plot adjustments

sns.set_style('darkgrid')
sns.set_context('talk')
sns.set_palette('deep')

## Now, it is evident that Total Spent is a determining variable for 
## units sold (or Quantity), as seen by:
    
## Online
print(np.corrcoef(online['Total Spent'], online.Quantity)) ## 0.92

sns.jointplot(data = online, x = 'Total Spent', y= 'Quantity',
            kind='reg')
plt.title('Scatter Plot Online Total Spent and Quantity', weight = 'bold',
          fontsize = 16)
plt.xlabel('Total Spent per Month', fontsize = 12)
plt.ylabel('Units Sold Per month', fontsize = 12, labelpad=10)
plt.tight_layout()
plt.show()

## In-store
print(np.corrcoef(InStore['Total Spent'], InStore.Quantity)) ## 0.93

sns.jointplot(data = InStore, x = 'Total Spent', y= 'Quantity',
            kind='reg')
plt.title('Scatter Plot Online Total Spent and Quantity', weight = 'bold',
          fontsize = 16)
plt.xlabel('Total Spent per Month', fontsize = 12)
plt.ylabel('Units Sold Per month', fontsize = 12, labelpad=10)
plt.tight_layout()
plt.show()


## Due to this strong correlation and Quantity being a count variable,
## OLS and Poisson regressions will be used as models.


## OLS regression

## Online

## Preparing data:
    
## Selecting explanatory variables and adding intercept    
x_matrix_online = online[['Total Spent', 'MonthInd', 'Season']]
x_matrix_online = sm.add_constant(x_matrix_online)

## Selecting response variable
y_online = online.Quantity

## Fitting model:

ols_online = sm.OLS(y_online, x_matrix_online).fit()
print(ols_online.summary())
## The probability of the resulting F-statistic (3.06e-139) and an R2 of 0.896
## demonstrate that this model is predicting Quantity very accurately based
## on the parameters given, and that it is extremely unlikely that the model
## is guessing correctly but uneducatedly in this situation. This
## indicates that the model is highly reliable and effective.

## Predicting values:

online_predictions = ols_online.predict(x_matrix_online)

## Validating model:
    
residuals_ols_o = y_online - online_predictions

## The following plot shows that residual values do not follow a specific trend,
## and are centered around 0, varying from -0.4 to 0.5 (estimates). This
## reflects that the model is free of issues.
plt.scatter(x=y_online, y=residuals_ols_o, alpha=0.7)
plt.axhline(0, color='black', linestyle='--')
plt.title('Residual Plot OLS Online (log-scale)', fontsize=16, weight = 'bold')
plt.xlabel('Fitted log-Quantity', fontsize=12)
plt.ylabel('Residual (log-scale)', fontsize=12)
plt.tight_layout()
plt.show()

## Again, very healthy and powerful metrics for prediction.
print("Online evaluation metrics (rmse, mae, r2):", 
      regression_metrics(y_online, online_predictions))


## In-store:
    
x_matrix_InStore = InStore[['Total Spent', 'MonthInd', 'Season']]
x_matrix_InStore = sm.add_constant(x_matrix_InStore)

y_InStore = InStore.Quantity

ols_InStore = sm.OLS(y_InStore, x_matrix_InStore).fit()
print(ols_InStore.summary())
## The probability of the resulting F-statistic (2.02e-143) and an R2 of 0.903
## demonstrate that this model is predicting Quantity very accurately based
## on the parameters given, and that it is extremely unlikely that the model
## is guessing correctly but uneducatedly in this situation. This
## indicates that the model is highly reliable and effective.

InStore_predictions = ols_InStore.predict(x_matrix_InStore)

residual_ols_I = y_InStore - InStore_predictions

## The following plot shows that residual values do not follow a specific trend,
## and are centered around 0, varying from -0.4 to 0.5 (estimates). This
## reflects that the model is free of issues. Mostly the same plot as for
## online data.
plt.scatter(x=y_InStore, y=residual_ols_I, alpha=0.7)
plt.axhline(0, color='black', linestyle='--')
plt.title('Residual Plot OLS In-store (log-scale)', fontsize=16, weight = 'bold')
plt.xlabel('Fitted log-Quantity', fontsize=12)
plt.ylabel('Residual (log-scale)', fontsize=12)
plt.tight_layout()
plt.show()

## Again, very healthy and powerful metrics for prediction.
print("In-store evaluation metrics (rmse, mae, r2):", 
      regression_metrics(y_InStore, InStore_predictions))


## Poisson Regression:
    
## exp() log-transformed variables:
    
online.Quantity = np.exp(online.Quantity)
online['Total Spent'] = np.exp(online['Total Spent'])
InStore.Quantity = np.exp(InStore.Quantity)
InStore['Total Spent'] = np.exp(InStore['Total Spent'])


## Ensuring equidispersion:
    
print(np.var(online.Quantity)/np.mean(online.Quantity))
print(np.var(InStore.Quantity)/np.mean(InStore.Quantity))

## This ratios suggest that using Poisson distributions and models is not
## a good fit. They will not be used.

joblib.dump(ols_online, 'OLS online')
joblib.dump(ols_InStore, 'OLS InStore')
