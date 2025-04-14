# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 20:17:11 2025

@author: nicol
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
import joblib

np.random.seed(42)

online = pd.read_pickle("C:/Users/nicol/OneDrive/Desktop/Bayesian Regression on Profit/online.pkl")
InStore = pd.read_pickle("C:/Users/nicol/OneDrive/Desktop/Bayesian Regression on Profit/InStore.pkl")


## Functions needed.

def log_likelihood(y, y_pred, sigma=10):
    return -0.5 * np.sum(((y - y_pred) / sigma)**2)

def log_prior(beta):
    return -0.5 * np.sum(beta**2)

def tracePlot(betas):
    
    names = ['Intercept', 'Total Spent', 'MonthInd', 'Season']
    
    fig, axes = plt.subplots(2, 2, figsize = (8, 10))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        sns.lineplot(x=range(1000), y=betas[:,i], ax=ax)
        ax.set_title(names[i])

    fig.suptitle('Trace Plots', weight = 'bold', fontsize = 16)
    fig.supxlabel('Index', fontsize=12)
    fig.supylabel('Value', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    
def regression_metrics(actual, predicted):
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    mae = np.mean(np.abs(predicted - actual))
    r2 = 1 - np.sum((actual - predicted)**2) / np.sum((actual - np.mean(actual))**2)
    return rmse, mae, r2
    

        ## Manual NumPy implementation

## General variables

n_iter = 1000

## Online location

y_online = online.Quantity.values ## Actual Y to compare to
x_online = online[['Total Spent', 'MonthInd', 'Season']].copy() ## Copy of values in X
x_online['Intercept'] = 1 ## Adding intercept
x_online_matrix = x_online[['Intercept', 'Total Spent', 'MonthInd', 'Season']].values ## Final Matrix
## This input ensures no leakage from the original data's Quantity


n_features = x_online_matrix.shape[1] ## How many variables?
betas_online = np.zeros((1000, n_features)) ## 4 features per 1000 repetitions
beta_current_online = np.random.normal(0, 1, n_features)



for i in range(n_iter):
    
    ## Proposing a set of weights (betas)
    proposed_beta = beta_current_online + np.random.normal(0, 0.05, n_features)
    
    ## Calculating Y's from current and proposal
    
    y_from_current = x_online_matrix @ beta_current_online
    y_from_proposal = x_online_matrix @ proposed_beta
    
    ## Now that both results are availiable, lets compare them.
    ## We will generate tentative posterior data, done by likelyhood + prior
    
    ## We need the comparisson of the generated y and the actual one, and the
    ## information from our proposed set of weights (betas)
    
    log_posterior_current = log_likelihood(y_online, y_from_current) + log_prior(beta_current_online)
    log_posterior_proposed = log_likelihood(y_online, y_from_proposal) + log_prior(proposed_beta)
    
    ## Now, having both probabilities of success for each Y, calculate acceptance P
    
    accept_prob = min(1, np.exp(log_posterior_proposed - log_posterior_current))
    
    ## With that value, decide if the proposed set of betas is fit enough
    ## to be considered. If so, set it as the new prior (or current).
    
    if np.random.rand() < accept_prob:
        beta_current_online = proposed_beta
    
    ## Independently of which one is selected, current or proposed, add it
    ## to the storage.
    
    betas_online[i] = beta_current_online
    
    ## Done.
    

## InStore Location, no explanation comments
    
y_InStore = InStore.Quantity.values
x_InStore = InStore[['Total Spent', 'MonthInd', 'Season']].copy()
x_InStore['Intercept'] = 1
x_InStore_matrix = x_InStore[['Intercept', 'Total Spent', 'MonthInd', 'Season']].values

n_features_InStore = x_InStore_matrix.shape[1]
betas_InStore = np.zeros((1000, n_features_InStore))
beta_current_InStore = np.random.normal(0, 1, n_features_InStore)

for i in range(n_iter):
    
    proposed_beta = beta_current_InStore + np.random.normal(0, 0.05, n_features_InStore)
    
    y_from_current = x_InStore_matrix @ beta_current_InStore
    y_from_proposal = x_InStore_matrix @ proposed_beta
    
    log_posterior_current = log_likelihood(y_InStore, y_from_current) + log_prior(beta_current_InStore)
    log_posterior_proposed = log_likelihood(y_InStore, y_from_proposal) + log_prior(proposed_beta)
    
    accept_prob = min(1, np.exp(log_posterior_proposed - log_posterior_current))
    
    if np.random.rand() < accept_prob:
        beta_current_InStore = proposed_beta
        
    betas_InStore[i] = beta_current_InStore
    

## Before removing the burn-in stage, model convergence will be assessed
## through trace plots.


    
    
## The following plots verify that the Markov Chain has converged, validating 
## future validation measures. They are highly variable at the beginning, but
## stabilize as repetitions increase.
tracePlot(betas_online)
tracePlot(betas_InStore)

## Now that there are the samples for both locations, delete burn-in period.
## This leaves records from after the Markov Chain had converged.
burn_in = 200
betas_online = betas_online[burn_in:]
print(betas_online.shape) ## From 1000 to 800 records
betas_InStore = betas_InStore[burn_in:]
print(betas_InStore.shape) ## From 1000 to 800 records


## Distributions for every variable's weight for regression

variables = ['Intercept', 'Total Spent', 'MonthInd', 'Season']

## Online orders:
    
fig, axes = plt.subplots(2, 2, figsize=(8, 10))

for i, ax in enumerate(axes.flat):
    sns.histplot(betas_online[:,i], kde = True, ax=ax, color='skyblue')
    ax.set_title(variables[i])
    
fig.suptitle('Posterior Distribution for Betas (online)', fontsize = 16)
fig.supxlabel('Value', fontsize=12)
fig.supylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

## In-store orders:
    
fig, axes = plt.subplots(2, 2, figsize=(8, 10))

for i, ax in enumerate(axes.flat):
    sns.histplot(betas_InStore[:,i], kde = True, ax=ax, color='skyblue')
    ax.set_title(variables[i])
    
fig.suptitle('Posterior Distribution for Betas (in-store)', fontsize = 16)
fig.supxlabel('Value', fontsize=12)
fig.supylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()


## Results and evaluation.
    


## Online:

## Actual Quantities
stocks_online = online.Quantity

## Evaluation through frequentist approach: RMSE, MAE, R**2

## For this, we will use the median betas from each storage and calculate 
## the resulting Y's for every availiable record, to use the appropriate metrics
## to compare them.

median_betas_online = np.median(betas_online, axis = 0)
predicted_stocks_online = x_online_matrix @ median_betas_online.T

print("Online evaluation metrics (rmse, mae, r2):", 
      regression_metrics(stocks_online, predicted_stocks_online))

## InStore:
    
stocks_InStore = InStore.Quantity

median_betas_InStore = np.median(betas_InStore, axis = 0)
predicted_stocks_InStore = x_InStore_matrix @ median_betas_InStore.T

print("In-store evaluation metrics (rmse, mae, r2):", 
      regression_metrics(stocks_InStore, predicted_stocks_InStore))


## Given these metrics, it is evident that the predictions for online stock
## are not accurate enough, but the in-store storages do align with the model's
## predictions, as it is able to explain 77.5% of the stock amounts. This was
## expected. From the EDA, it is observable that quantities ordered from in-store
## are significantly more stable than online, explaining the differences in
## model performance.


        ## PyMC implementations.
        
## Online location

with pm.Model() as model_o:
    
    
    ## Prior distribution's values for beta and sigma. Betas derived from
    ## N(0, 1), and sigma coming from pm.HalfNormal to ensure sigma > 0.
    beta = pm.Normal('beta', mu=0, sigma=1, shape=x_online_matrix.shape[1])
    sigma = pm.HalfNormal('sigma', 1)


    ## Predicted values of Y with current set of betas (same as 
    ## x_online_matrix @ beta)
    predicted_y = pm.math.dot(x_online_matrix, beta)
    
    ## Likelihood distribution
    likelihood = pm.Normal('likelihood', mu=predicted_y, sigma=sigma, 
                           observed=y_online)
    
## Execute iterations with MH algorithm
with model_o:
    trace = pm.sample(1000, step = pm.Metropolis(vars=[beta], scaling=0.05) , 
                      random_seed=42, chains=1, cores=1, tune=200)

pymc_o_betas = trace.posterior['beta'].values.squeeze(axis=0)

## Trace plots:
## These trace plots do not show optimal trends, but the most significant 
## parameters explore various ranges in the Y direction, while still having 
## variance but following a general trend. Still, they are appropriate enough
## to conclude that the model has converged.
tracePlot(pymc_o_betas)


## InStore location

with pm.Model() as model_I:
    
    beta = pm.Normal('beta', mu = 0, sigma = 1, shape=x_InStore_matrix.shape[1])
    
    sigma = pm.HalfNormal('sigma', 1)
    
    predicted_y = pm.math.dot(x_InStore_matrix, beta)
    
    likelihood = pm.Normal('Likelihood', mu=predicted_y, sigma=sigma,
                           observed=y_InStore)
    
with model_I:
    traceI = pm.sample(1000, step = pm.Metropolis(vars=[beta], scaling=0.05) , 
                      random_seed=42, chains=1, cores=1, tune=200)
    
pymc_I_betas = traceI.posterior['beta'].values.squeeze(axis=0)

## Likewise, these plots show traces that follow a general trend, but still
## explore the y axis, and have variance within the trend.
tracePlot(pymc_I_betas)


## Results and evaluation.

## Online:

median_betas_pyo = np.median(pymc_o_betas, axis=0)
predicted_stocks_pyo = x_online_matrix @ median_betas_pyo.T

median_betas_pyI = np.median(pymc_I_betas, axis=0)
predicted_stocks_pyI = x_InStore_matrix @ median_betas_pyI

print("Online evaluation metrics (rmse, mae, r2):", 
      regression_metrics(stocks_online, predicted_stocks_pyo))
print("In-store evaluation metrics (rmse, mae, r2):", 
      regression_metrics(stocks_InStore, predicted_stocks_pyI))

## As shown, these models are able to explain 80% and 73.5% of
## the stocks needed for the online and in-store locations respectively,
## making them highly reliable for stock optimization. Still, the manual
## implementation of the HM algorithm sliglty outperforms the PyMC version,
## which will be taken into account for future steps.


## The best fitting models will be saved for usage.

joblib.dump(betas_InStore, 'bayesian_betas_InStore.pkl')
joblib.dump(pymc_o_betas , 'bayesian_betas_Online.pkl')