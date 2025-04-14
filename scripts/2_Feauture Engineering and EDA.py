# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 11:01:38 2025

@author: nicol
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

transactions = pd.read_pickle("C:/Users/nicol/OneDrive/Desktop/Bayesian Regression on Profit/transactions_clean.pkl")
transactions.dtypes


        ## Feauture engnineering 1
        
## The purpose of this project is to optimize stock per month per location, so
## engineer accordingly.


## Adding columns

transactions['Year'] = transactions['Transaction Date'].dt.year
transactions['Month'] = transactions['Transaction Date'].dt.month

## Removing Columns

## Not needed for specific purposes:
transactions.drop(columns=['Transaction ID', 'Customer ID', 'Item', 'Price Per Unit',
                           'Payment Method','Discount Applied'], inplace = True)
## Features already extracted -> redundant:
transactions.drop('Transaction Date', axis = 1, inplace = True)

## Removing January 2025, since it is up to the 18th and will affect data.

transactions = transactions.loc[transactions.Year != 2025, :]

## Aggregating data for modelling purposes:
    
transactions = transactions.groupby(['Category', 'Location', 'Year',
                                     'Month']).agg({'Quantity':'sum',
                                                    'Total Spent':'sum'}).reset_index()

                ## EDA
                
## This process evaluates the trends in Quantities ordered across time, from
## 2022 to 2024, and dove into where the purchases are being made, in order
## to optimize stock location. It shows insights on total units sold per
## location, late trends on the shift of location needs, and others.

                                            
## Range of dates from the data

print(transactions.Year.min())
print(transactions.Year.max())
## Spanning from 2022 to 2024.

## Plots:
sns.set_style('darkgrid')
sns.set_context('talk')
sns.set_palette('deep')

## Revenue per year, roughly equal.
yearly_revenue = transactions.groupby('Year')['Total Spent'].sum().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(x = 'Year', y='Total Spent', data = yearly_revenue,
            linewidth = 1, edgecolor = 'black')
plt.title('Revenue by Year', weight = 'bold', fontsize = 16)
plt.xlabel('Year', fontsize = 12)
plt.ylabel('Revenue', fontsize = 12, labelpad = 10)
plt.tight_layout()
plt.show()

## Monthly revenue:
## Unusual trends: high spending in January, irregular trends between years,
## slow period in December 2022. This urges us to revise Quantity sold.

monthly_revenue = transactions.groupby(['Year', 'Month'])['Total Spent'].sum().reset_index()
monthly_revenue = monthly_revenue.sort_values('Month')
plt.figure(figsize=(8,5))
sns.lineplot(x='Month', y='Total Spent', hue='Year', data=monthly_revenue)
plt.title('Monthly Revenue by Year', weight='bold', fontsize=16)
plt.xlabel('Month (1-12)', fontsize=12)
plt.ylabel('Revenue', fontsize=12, labelpad=10)
plt.tight_layout()
plt.show()

## Quantities sold:

monthly_quantities = transactions.groupby(['Year', 'Month']).Quantity.sum().reset_index()
monthly_quantities = monthly_quantities.sort_values('Month')
sns.lineplot(x='Month', y='Quantity', hue='Year', data=monthly_quantities)
plt.title('Monthly Sold Units by Year', weight='bold', fontsize=16)
plt.xlabel('Month (1-12)', fontsize=12)
plt.ylabel('Units Sold', fontsize=12, labelpad=10)
plt.tight_layout()
plt.show()

## Similarities between these 2 plots show that surges and dips in revenue
## do depend on product demand, not selection of products.


## Are more units being sold online, or in-store?
## Roughly the same amount.
quantity_per_location = transactions.groupby('Location').Quantity.sum().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(x='Location', y='Quantity', data=quantity_per_location
            , estimator=sum, linewidth=1, edgecolor='black')
plt.title('Units Sold by Location', weight='bold', fontsize=16)
plt.xlabel('Location', fontsize=12)
plt.ylabel('Total Units', fontsize=12, labelpad=10)
plt.tight_layout()
plt.show()

## Is there any trend of Location change lately?
## Online sales have a higher slope and value in recent years, indicating
## a current and future need to keep units stored for online purchases the most.
timeQuantity_per_location = transactions.groupby(['Year', 'Location']).Quantity.sum().reset_index()
sns.lineplot(x='Year', y='Quantity', hue='Location', data=timeQuantity_per_location)
plt.title('Storage Needed per Year', weight='bold', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Units', fontsize=12, labelpad=10)
plt.tight_layout()
plt.show()

## Demand per location, by month of the year.
## Only January shows more stock need in store over online.
timeDemand_location = transactions.groupby(['Location', 'Month']).Quantity.sum().reset_index()
timeDemand_location = timeDemand_location.sort_values('Month')
sns.lineplot(x='Month', y='Quantity', hue='Location', data=timeDemand_location)
plt.title('Monthly Quantities Ordered per Location', weight='bold', fontsize=16)
plt.xlabel('Month (1-12)', fontsize=12)
plt.ylabel('Quantity', fontsize=12, labelpad = 10)
plt.tight_layout()
plt.show()

## Total Units Ordered Across time:

monthly_overall = transactions.groupby(['Year', 'Month']).Quantity.sum().reset_index()
monthly_overall['Date'] = pd.to_datetime(monthly_overall[['Year', 'Month']].assign(DAY=1))
monthly_overall = monthly_overall.sort_values('Date')

sns.lineplot(x='Date', y='Quantity', data=monthly_overall)
plt.title('Total Units Ordered Over Time', weight='bold', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Units Sold', fontsize=12, labelpad=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## Quantities sold distribution
## This shows a skewed distribution, will need fixing.
sns.histplot(x=transactions.Quantity, bins=30, linewidth=1, edgecolor='black')
plt.title('Quantity Distribution')
plt.xlabel('Unit Amount', fontsize=12)
plt.ylabel('Frequency', fontsize=12, labelpad=10)
plt.tight_layout()
plt.show()
print("Distribution Kurtosis:", kurtosis(transactions.Quantity)) ## 1.32
print("Distribution Skewness:", skew(transactions.Quantity))     ## 1.29
## This describes a distribution significatly skewed to the right, and one 
## with longer tails that normal.

## Total Spent distribution
sns.histplot(x=transactions['Total Spent'], bins=30, linewidth=1, edgecolor='black')
plt.title('Total Spent Distribution')
plt.xlabel('Total Spent', fontsize=12)
plt.ylabel('Frequency', fontsize=12, labelpad=10)
plt.tight_layout()
plt.show()
print("Distribution Kurtosis:", kurtosis(transactions['Total Spent'])) ## 0.32
print("Distribution Skewness:", skew(transactions['Total Spent']))     ## 0.85
## Due to skewness, this distribution will also need fixing. It will be used
## as an estimator for Quantity.

        ## Feature Engineering 2
        
transactions.Quantity = np.log(transactions.Quantity)
## See a now normal distribution 
sns.histplot(x=transactions.Quantity, bins=30, linewidth=1, edgecolor='black')
plt.title('Quantity Distribution')
plt.xlabel('Unit Amount', fontsize=12)
plt.ylabel('Frequency', fontsize=12, labelpad=10)
plt.tight_layout()
plt.show()
print("Distribution Kurtosis:", kurtosis(transactions.Quantity)) ## 0.26
print("Distribution Skewness:", skew(transactions.Quantity))     ## -0.27
## Kurtosis reduced from 1.32 to 0.26, skewness reduced from 1.29 to -0.27

transactions['Total Spent'] = np.log(transactions['Total Spent'])
## Also a normal distribution.
sns.histplot(x=transactions['Total Spent'], bins=30, linewidth=1, edgecolor='black')
plt.title('Total Spent Distribution')
plt.xlabel('Total Spent', fontsize=12)
plt.ylabel('Frequency', fontsize=12, labelpad=10)
plt.tight_layout()
plt.show()
print("Distribution Kurtosis:", kurtosis(transactions['Total Spent'])) ## 0.93
print("Distribution Skewness:", skew(transactions['Total Spent']))     ## -0.53

## Dividing data per location for separate modeling
online = transactions.loc[transactions.Location == 'Online', :]
InStore = transactions.loc[transactions.Location == 'In-store', :]


## Marking months of demand surge per location data
## We see from the EDA that online, January, July, November and December
## experience high demand or demand surges, as well as January, June, October
## and December for In-store purchases. Encode accordingly.

online['Season'] = 0
InStore['Season'] = 0

def encodeHighSeason(month, data):
    data.loc[data.Month == month, 'Season'] = 1
    
encodeHighSeason(1, online)
encodeHighSeason(7, online)
encodeHighSeason(11, online)
encodeHighSeason(12, online)

encodeHighSeason(1, InStore)
encodeHighSeason(6, InStore)
encodeHighSeason(10, InStore)
encodeHighSeason(12, InStore)


## Adding a month index to make linearity accessible to the model:

online.head() ## See datasets already sorted by date, ascending.
InStore.head()

online['MonthInd'] = np.arange(288)
InStore['MonthInd'] = np.arange(288)

## Feature Engineering and EDA completed.

online.to_pickle("online.pkl")
InStore.to_pickle("InStore.pkl")


