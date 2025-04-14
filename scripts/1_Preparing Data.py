# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 15:40:57 2025

@author: nicol
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

transactions = pd.read_csv('C:/Users/nicol/OneDrive/Desktop/Bayesian Regression on Profit/retail_store_sales.csv')


        ## Handling Missing Values

transactions.isna().sum()

## Imputing is based on dataset description found on Kaggle.


## Product Related missing values

len(transactions.Item.unique()) ## 201 different items
transactions.Item.unique()

def fillPriceByItem(itemCode):
    itemInfoDF = transactions.loc[(transactions.Item == itemCode) 
                                  & ~(transactions['Price Per Unit'].isna()), :]
    if itemInfoDF.empty:
        return
    
    fullInfoRow = itemInfoDF.iloc[0]
    
    transactions.loc[transactions.Item == itemCode, 'Price Per Unit'] = fullInfoRow['Price Per Unit']

for item in transactions.Item.unique():
    fillPriceByItem(item)
    
    
## Filling Item based on Price

def fillItemByPrice(price):
    itemInfoDF = transactions.loc[(transactions['Price Per Unit'] == price) 
                                  & ~(transactions.Item.isna()), :]
    if itemInfoDF.empty:
        return
    
    fullInfoRow = itemInfoDF.iloc[0]
    
    transactions.loc[transactions['Price Per Unit'] == price, 'Item'] = fullInfoRow.Item

for price in transactions['Price Per Unit'].unique():
    fillItemByPrice(price)
    
## Second run of fillPriceByItem needed to re fill based on newly added Item records
    
for item in transactions.Item.unique():
    fillPriceByItem(item)
    
    
## Revenue Related Missing values

## Rows with 2 values out of Price Per Unit, Quantity and Total Spent can be 
## filled, if 2 are missing, they should be dropped.

transactions.drop(
    transactions[transactions['Price Per Unit'].isna() & transactions.Quantity.isna()].index,
    axis = 0, 
    inplace = True)
transactions.drop(
    transactions[transactions['Price Per Unit'].isna() & transactions['Total Spent'].isna()].index,
    axis = 0, 
    inplace = True)
transactions.drop(
    transactions[transactions['Total Spent'].isna() & transactions.Quantity.isna()].index,
    axis = 0, 
    inplace = True)


transactions.isna().sum()
## Now, since there are no missing values for Quantity or Total Spent, all Prices
## Per Unit can be filled

transactions.loc[transactions['Price Per Unit'].isna(), 
                 'Price Per Unit'] = transactions['Total Spent']/transactions.Quantity


transactions.isna().sum()
## This leaves no missing values for Price Per Unit, allowing to fill in Item
## by price again.

for price in transactions['Price Per Unit'].unique():
    fillItemByPrice(price)


## Filling "Discount Applied".
## Filling missing discount values as False allows the dataset to keep the
## healthy variance introduced by discounts, but also to adjust the inconsistent
## values where Total Spent is miscalculated with no discount.

transactions.loc[transactions['Discount Applied'].isna(), 'Discount Applied'] = False


## Final check for missing values:
print("Total missing values: ")
print(transactions.isna().sum().sum())



        ## Memory usage and operation speed

transactions.dtypes
initialMemory = transactions.memory_usage(deep = True).sum()
transactions.memory_usage(deep = True)

## From the columns of type object, we ackowledge:

## 11971 unique values for Transaction ID
len(transactions['Transaction ID'].unique())
## 25 unique values for Customer ID
len(transactions['Customer ID'].unique())
## 8 unique values for Category
len(transactions.Category.unique())
## 25 unique values for Item
len(transactions.Item.unique())
## 3 unique values for Payment Method
len(transactions['Payment Method'].unique())
## 2 unique values for Location
len(transactions.Location.unique())
## 2 unique values for Discount Applied
len(transactions['Discount Applied'].unique())
## Ignore Transaction Date, type will be adjusted eventually.

## This indicates that all values, except for Transaction ID and Date can be
## casted as category.

transactions['Customer ID'] = transactions['Customer ID'].astype('category')
transactions.Category = transactions.Category.astype('category')
transactions.Item = transactions.Item.astype('category')
transactions['Payment Method'] = transactions['Payment Method'].astype('category')
transactions.Location = transactions.Location.astype('category')
transactions['Discount Applied'] = transactions['Discount Applied'].astype('category')

newMemory = transactions.memory_usage(deep = True).sum()
print("Memory usage reduced to:")
print(newMemory/initialMemory*100, "percent.")
print("Memory usage reduced from", initialMemory, "to", newMemory, "(bytes)")
## This resulted in around 65% memory usage reduction.


        ## Handling date data type.
        
transactions['Transaction Date'] = pd.to_datetime(transactions['Transaction Date'])
transactions.dtypes


        ## Removing outliers

## 2 types of outliers are focused: Total Spent outliers; will be addressed
## with high flexibility since high ticket items are legitimaletely bought 
## occasioanlly, Quantity outliers; purchases with too many items purchased
## are more likely to be real outliers that affect the dataset.

## The following function will be a helper for removal.

def remove_outliers(column, low, high):
    return transactions.loc[(transactions[column] > low) 
                            & (transactions[column] < high), :]

## Total Spent outliers:
    
transactions['Total Spent'].skew() ## 0.83
transactions['Total Spent'].kurtosis() ## -0.10 (2.9)

## This combination of moderate skewness, and low kurtosis, allows to remove
## outliers using z-score, but requires a low threashold. Therefore, 2sd
## will be used.

totalSpentSd = np.std(transactions['Total Spent'])
totalSpentMean = transactions['Total Spent'].mean()
transactions = remove_outliers('Total Spent', totalSpentMean - 2*totalSpentSd,
                              totalSpentMean + 2*totalSpentSd)

## Quantity outliers:
    
transactions.Quantity.skew() ## 0.05
transactions.Quantity.kurtosis() ## -1.15 (1.85)
transactions.Quantity.max() ## 10

## These measures indicate that there are no significant outliers from this 
## class, and it will be left as is.



        ## Fixing and checking for inconsistencies in the data

## 0 columns like:
transactions.loc[(transactions['Discount Applied'] == False) & 
                 (transactions['Total Spent'] != (transactions.Quantity *
                                                  transactions['Price Per Unit']))].shape[0]

## Latest transaction: from a past date.
transactions['Transaction Date'].max()

##
transactions.Category.unique().shape ## 8 item categories

## The following function will check if every product id is matched to its
## corresponding category

def match_category(category, item_indicator):
    mismatches = transactions.loc[
        (transactions.Item.str.contains(item_indicator)) &
        (transactions.Category != category)]
    return mismatches.shape[0]
    

match_category('Patisserie', 'PAT') ## 508 mismatches
transactions.loc[transactions.Item.str.contains('PAT'), 'Category'] = 'Patisserie'
match_category('Patisserie', 'PAT') ## Fixed

match_category('Milk Products', 'MILK') ## 783 mismatches
transactions.loc[transactions.Item.str.contains('MILK'), 'Category'] = 'Milk Products'
match_category('Milk Products', 'MILK') ## Fixed

match_category('Butchers', 'BUT') ## 1612 mismatches
transactions.loc[transactions.Item.str.contains('BUT'), 'Category'] = 'Butchers'
match_category('Butchers', 'BUT') ## Fixed

match_category('Beverages', 'BEV') ## 1252 mismatches
transactions.loc[transactions.Item.str.contains('BEV'), 'Category'] = 'Beverages'
match_category('Beverages', 'BEV') ## Fixed

match_category('Food', 'FOOD') ## 2870 mismatches
transactions.loc[transactions.Item.str.contains('FOOD'), 'Category'] = 'Food'
match_category('Food', 'FOOD') ## Fixed

match_category('Furniture', 'FUR') ## 721 mismatches
transactions.loc[transactions.Item.str.contains('FUR'), 'Category'] = 'Furniture'
match_category('Furniture', 'FUR') ## Fixed

match_category('Electric household essentials', 'EHE') ## 1055 mismatches
transactions.loc[transactions.Item.str.contains('EHE'), 'Category'] = 'Electric household essentials'
match_category('Electric household essentials', 'EHE') ## Fixed

match_category('Computers and electric accessories', 'CEA') ## 827 mismatches
transactions.loc[transactions.Item.str.contains('CEA'), 'Category'] = 'Computers and electric accessories'
match_category('Computers and electric accessories', 'CEA') ## Fixed



        ## Duplicates
        
print('Amount of duplicated records:', transactions.duplicated().sum()) ## No duplicates.


## Data Preprocessing Finished.
transactions.dtypes
transactions.to_pickle("transactions_clean.pkl")
