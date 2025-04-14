# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 12:51:42 2025

@author: nicol
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import joblib


## The models will be tested on this other dataset from kaggle.
## https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset
## This dataset has a structure of more expenditure compared to quantity
## in relation to the other data, which will likely result in reduced performance.
## These flaw will likely cause the model to estimate higher needs for stock
## than the Quantities needed on the dataset.

## Functions:
    
def regression_metrics(actual, predicted):
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    mae = np.mean(np.abs(predicted - actual))
    r2 = 1 - np.sum((actual - predicted)**2) / np.sum((actual - np.mean(actual))**2)
    return rmse, mae, r2


def predict_bayesian(data, low, high):
    
    y_low = data @ low.T
    y_high = data @ high.T
    
    return y_low, y_high


## Preparing data:

salesTEST = pd.read_csv("my_file_path")
## Reducing sample:
salesTEST = salesTEST.loc[0:1999, :]
salesTEST.shopping_mall.value_counts()
## Dropping unnecessary column
salesTEST.drop('invoice_no', axis = 1, inplace = True)

## Kanyon and Mall of Istanbul will be assumed as InStore arbitrarily.

salesTEST['Location'] = 'Online'
salesTEST.loc[salesTEST.shopping_mall.isin(['Kanyon', 'Mall of Istanbul']),
                                           'Location']  = 'In-store'

salesTEST.drop('shopping_mall', axis = 1, inplace = True)

## Handling dates
salesTEST.invoice_date = pd.to_datetime(salesTEST.invoice_date, format='%d/%m/%Y')
salesTEST['Year'] = salesTEST.invoice_date.dt.year
salesTEST['Month'] = salesTEST.invoice_date.dt.month
salesTEST.drop('invoice_date', axis = 1, inplace = True)

## Ordering chronologically

salesTEST.sort_values(['Year', 'Month'], inplace=True)

## Grouping data

salesTEST = salesTEST.groupby(['Year', 'Month', 
                               'Location'])[['quantity', 'price']].sum().reset_index()

## Renaming and engineering to match train data

salesTEST.rename(columns={'quantity':'Quantity', 'price':'Total Spent'},
                 inplace = True)

onlineTEST = salesTEST.loc[salesTEST.Location == 'Online', :]
InStoreTEST = salesTEST.loc[salesTEST.Location == 'In-store', :]

onlineTEST['Season'] = 0
InStoreTEST['Season'] = 0
def encodeHighSeason(month, data):
    data.loc[data.Month == month, 'Season'] = 1
    
encodeHighSeason(1, onlineTEST)
encodeHighSeason(7, onlineTEST)
encodeHighSeason(11, onlineTEST)
encodeHighSeason(12, onlineTEST)

encodeHighSeason(1, InStoreTEST)
encodeHighSeason(6, InStoreTEST)
encodeHighSeason(10, InStoreTEST)
encodeHighSeason(12, InStoreTEST)

onlineTEST['MonthInd'] = np.arange(27)
InStoreTEST['MonthInd'] = np.arange(27)



onlineTEST.Quantity = np.log(onlineTEST.Quantity)
onlineTEST['Total Spent'] = np.log(onlineTEST['Total Spent'])
InStoreTEST.Quantity = np.log(InStoreTEST.Quantity)
InStoreTEST['Total Spent'] = np.log(InStoreTEST['Total Spent'])

## Setting actual Y's

quantityOnline = onlineTEST.Quantity
quantityInStore = InStoreTEST.Quantity
onlineTEST.drop(columns=['Year', 'Month', 'Location'], axis = 1, inplace = True)
InStoreTEST.drop(columns=['Year', 'Month', 'Location'], axis = 1, inplace = True)

## Replicating model training structure

onlineTEST['Intercept'] = 1
InStoreTEST['Intercept'] = 1

onlineTEST = onlineTEST[['Intercept', 'Total Spent', 'MonthInd', 'Season']]
InStoreTEST = InStoreTEST[['Intercept', 'Total Spent', 'MonthInd', 'Season']]


## Loading models

ols_online = joblib.load("my_file_path")
ols_InStore = joblib.load("my_file_path")

bayesian_online = joblib.load("my_file_path")
bayesian_InStore = joblib.load("my_file_path")

## OLS model usage:

## Predictions
online_predicted = ols_online.predict(onlineTEST)
InStore_predicted = ols_InStore.predict(InStoreTEST)

## Evaluating measures:
    
print("Online evaluation metrics (rmse, mae, r2):", 
      regression_metrics(quantityOnline, online_predicted))
print("In-store evaluation metrics (rmse, mae, r2):", 
      regression_metrics(quantityInStore, InStore_predicted))

## Evidently, the results do not reflect an useful model, due to the
## previously highlighted differences in the data. From the beginning,
## it was acknowledged that the new dataset reflected higher spending per unit
## than the training set. This file is only meant to demonstrate how to use
## these models.

## Bayesian model usage:
    
## Setting posterior betas for credible interval creation

## Online model:     
o_interceptlow = np.percentile(bayesian_online[:, 0], 2.5)
o_intercepthigh = np.percentile(bayesian_online[:, 0], 97.5)

o_totallow = np.percentile(bayesian_online[:, 1], 2.5)
o_totalhigh = np.percentile(bayesian_online[:, 1], 97.5)

o_monthlow = np.percentile(bayesian_online[:, 2], 2.5)
o_monthhigh = np.percentile(bayesian_online[:, 2], 97.5)

o_seasonlow = np.percentile(bayesian_online[:, 3], 2.5)
o_seasonhigh = np.percentile(bayesian_online[:, 3], 97.5)

## InStore model:
    
I_interceptlow = np.percentile(bayesian_InStore[:, 0], 2.5)
I_intercepthigh = np.percentile(bayesian_InStore[:, 0], 97.5)

I_totallow = np.percentile(bayesian_InStore[:, 1], 2.5)
I_totalhigh = np.percentile(bayesian_InStore[:, 1], 97.5)

I_monthlow = np.percentile(bayesian_InStore[:, 2], 2.5)
I_monthhigh = np.percentile(bayesian_InStore[:, 2], 97.5)

I_seasonlow = np.percentile(bayesian_InStore[:, 3], 2.5)
I_seasonhigh = np.percentile(bayesian_InStore[:, 3], 97.5)



online_low_betas = np.array([o_interceptlow, o_totallow, o_monthlow,
                             o_seasonlow])
online_high_betas = np.array([o_intercepthigh, o_totalhigh, o_monthhigh,
                              o_seasonhigh])
InStore_low_betas =np.array([I_interceptlow, I_totallow, I_monthlow,
                             I_seasonlow])
InStore_high_betas = np.array([I_intercepthigh, I_totalhigh, I_monthhigh,
                              I_seasonhigh])


online_matrix = onlineTEST.values
InStore_matrix = InStoreTEST.values

online_predicted_baylow, online_predicted_bayhigh = predict_bayesian(online_matrix, online_low_betas, online_high_betas)


InStore_predicted_baylow, InStore_predicted_bayhigh = predict_bayesian(InStore_matrix, InStore_low_betas, InStore_high_betas)


## To test these models, separate dataframes will be created with columns:
## Quantity, low limit and high limit, and the percetange of correct 
## predictions of the model will be calculated.

## Online:
    
bayesian_results_o = pd.DataFrame(quantityOnline)
online_low, online_high = predict_bayesian(online_matrix, online_low_betas
                                           , online_high_betas)

bayesian_results_o['Low'] = online_low
bayesian_results_o['High'] = online_high
bayesian_results_o['in_interval'] = 0
bayesian_results_o.loc[(bayesian_results_o.Low < bayesian_results_o.Quantity) 
                       & (bayesian_results_o.High > bayesian_results_o.Quantity),
                       'in_interval'] = 1

print('Online bayesian correct predictions:',
      bayesian_results_o.loc[bayesian_results_o.in_interval == 1,:].shape[0])


## In-store:
    
bayesian_results_I = pd.DataFrame(quantityInStore)
InStore_low, InStore_high = predict_bayesian(InStore_matrix, InStore_low_betas
                                           , InStore_high_betas)

bayesian_results_I['Low'] = InStore_low
bayesian_results_I['High'] = InStore_high
bayesian_results_I['in_interval'] = 0
bayesian_results_I.loc[(bayesian_results_I.Low < bayesian_results_I.Quantity) 
                       & (bayesian_results_I.High > bayesian_results_I.Quantity), 
                       'in_interval'] = 1

print('In-store bayesian correct predictions:',
      bayesian_results_I.loc[bayesian_results_I.in_interval == 1,:].shape[0])


## In the same manner as OLS models, the accuracy of the models is completely
## unreliable. Again, this is expected due to the differences between the
## data that was used to build the model and the new dataset.
## It is necessary to remark that this file is only supposed to be educative 
## on how to use these models, not designed to actually predict the stock 
## needs of the company of origin for the newly imported data.
