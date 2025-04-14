- Project title:

"Forecasting Commercial Stock Needs Through Manual and Automatic Bayesian and Frequentist approaches: a Comparison Between Libraries and Man-Made Models."


- Project Overview:

This project aims to follow a complete data science pipeline, from data cleaning to instructions for model replication, in order to forecast stock needs for retail stores. This project considers the following models: low-level and NumPy-implemented MCMC Metropolis Hastings algorithm, PyMC's implementation of MCMC Metropolis Hastings, OLS regression and Poisson regression from statsmodels.

- Business Problem:

Inventory allocation is a recurring challenge among retail stores. Companies must efficiently and correctly decide on how much stock to keep available for sale - in-store and online - without overstocking or understocking. Any of these will frequently result in revenue loss by the company, due to either missing the opportunity to generate more revenue, or leaving units unsold. 

This project aims to forecast the necessary monthly stock for companies, through the use of Bayesian and frequentist models usage and comparison. It also emphasizes real-world decision support by focusing on model interpretability and statistical support rather than purely accuracy metrics.

- Source data description:

The original dataset was taken from Kaggle, and consists of:

1. 12575 rows,
2. 11 Columns: Transaction ID, Customer ID, Category, Item, Price Per Unit, Quantity, Total Spent, Payment Method, Location, Transaction Date, Discount Applied,
3. 7229 missing values,
4. 6023316 bytes of memory usage,
5. No duplicated values and,
6. Data spanning from January 2022 to January 2025.

- Data Cleaning and Preprocessing:

See 'Preparing Data' file.

The following steps were taken within this stage:

1. Handling missing values by logically imputing columns. Among the practices used are: deriving missing values in Price Per Unit, Quantity and Total Spent when 2 values were available, filling Item based on Price Per Unit and vice versa, filling missing values for Discount Applied as False. Multiple parses of some of the implement practices were performed as new data became available (as values were filled).

2. Memory usage reduction for operation speed optimization. Categorical pandas series with low cardinality were transformed from Object type to Category, reducing memory usage by 65%, represented as a transformation from 6023316 to 2149085 bytes.

3. Casting Object pandas series Transaction Date as a datetime variable.

4. Removed outliers from Total Spent and Quantity variables, by margins derived from the skewness and kurtosis of every specific distribution. 622 rows were discarded through this process.

5. Eliminating inconsistencies in the data. The following inconsistencies were checked: Item (each Item has a code indicating its category) being synchronized with Category, rows containing a date posterior to the date of the construction, and rows with Applied Discount marked as false but with Total Spent not corresponding to Quantity * Price per unit.

6. Checking for duplicate rows, none were found.


- Exploratory Data Analysis and Feature Engineering.

See 'Feature Engineering and EDA' file.

This section was split into 3 stages. Feature Engineering 1 -> EDA -> Feature Engineering 2.

Feature engineering 1:

1. Adding relevant variables: 'Year' and 'Month' from Transaction Date.

2. Removing irrelevant or redundant variables: Transaction ID, Customer ID, Item, Price Per Unit, Payment Method, Discount Applied and Transaction Date.

3. Removing data from 2025, since it is only recorded until January 18. Keeping this data means including a year and month that is irregular from the pattern of Quantity and Total Spent, the most relevant variables for model construction.

4. Aggregating columns for modeling based on Category, Location, Year and Month, and deriving the totals of Quantity and Total Spent as the relevant data per group.

EDA (visual based):

1. Using max and min functions on Year, showing that data spans from 2022 to 2025 (once again).

2. Comparing revenue by year - bar plot. Values were roughly equal.

3. Comparing revenue by month - line plot, lines separated by year. Unusual trends of high spending in January, irregular trends between years at the same dates, slow period on Dec of 2022.

4. Comparing Quantity (units sold) by month - line plots, lines separated by year. Shows similar trends to the previous plot, indicating that irregularities in revenue are caused by total and not product-related demand shifts.

5. Comparing units sold per location, online or in-store - Bar plot. Values were roughly equal.

6. Identifying time trends on demand by location - line plot. Online demand is consistently higher, and posterior to a severe dip, it escalates more rapidly. In-store demand experiences slow but steady increases.

7. Comparing demand per location at every month - line plot. Only January experiences higher demand in-store than online.

8. Total Units ordered across time - line plot. Plot seems highly irregular, but traces of monthly trends can be seen.

9. Exploring Quantity distribution - histogram, skewness and kurtosis. The distribution is skewed to the right and with shorter tails than normal (kurtosis < 3).

10. Exploring Total Spent distribution - histogram, skewness and kurtosis. Slightly skewed but significantly shorter tails than normal

Both of these distributions will need transformation before modeling.

Feature engineering 2:

1. Log-transforming Quantity. A new histogram and skewness and kurtosis measures indicate lower kurtosis and slight skewness to the left. 

2. Log-transforming Total Spent. A new histogram and skewness and kurtosis measures indicate higher kurtosis and slight skewness to the left.

3. Dividing data per location into 2 sets: Online and Instore. This allows to optimize specific-location stock directly.

 4. Adding seasonality per location. Previous EDA revealed that more units were being sold on January, July, November and December online, and January, June, October and December in-store. A 'Season' variable was introduced, encoded to 1 on months with high demand or a demand surge and 0 otherwise.

5. Introducing a 'MonthInd' (month index) variable to allow linear modeling. This eliminates the need to divide the data be year in order to get linear interpretations and results. This variable was introduced using np.arange() and the length of the dataset as parameter.


Concluding this stage, the relevant features for the models to function are Quantity (target variable), Total Spent, Season and MonthInd. An 'Intercept' feature was later added in order to accommodate for the models.


- Bayesian Models implementations: both implementations follow a Metropolis Hastings algorithm to perform Bayesian regression.

See 'Bayesian Methods' file.

This section is divided in 2 main areas, manual NumPy implementation and high-level PyMC implementation. For both, a model for the online and InStore datasets was developed.

NumPy:

This algorithm uses a for loop to iterate through the specified number of iterations, using a NumPy matrix to store the predictive variables and an array to store the target values of Quantity. These are sorted to match the corresponding rows between both objects.

The set up is initially given this previously mentioned values, a storage matrix for the posterior sets of weights for the regression of every feature and a prior distribution of weights from a normal distribution.

Every iteration follows these steps:

1. Proposing a set of new weights coming from the prior distribution with added randomly generated noise. This allows dependency between every 2 consecutive suggested sets of weights (Markov Chain), as well as introducing randomness (Monte Carlo) in order to be able to explore diverse alternatives as the chain progresses.

2. Using matrix multiplication, Y values were obtained both from the current prior and the newly proposed sets of weights (of every iteration).

3. To compare the results, the log posterior probabilities the prior and proposed set of weights was calculated to later use to compute the acceptance probability for the specific iteration of the algorithm.

4. If the proposed posterior is accepted through the previous threshold, it is set as the new posterior. Otherwise, the prior remains as the posterior output, and therefore continues being the prior for the next iteration. In any case, the selected posterior is added to the storage to keep trace of the model's results.

This process was carried on for both locations.

PyMC:

1. using pm.Model(), a model from each location was created, containing betas randomly derived from a normal distribution in an array of the same length as the number of features (4), as well as a sigma value also randomly derived, but from a half-normal distribution, to ensure a positive value.

2. Predicted values of Y and likelihood distributions were calculated using pm.math.dot() and pm.Normal() with the 'observed' parameter included.

3. For every model (per location), traces were set to follow a length of 1000, using a Metropolis Hastings algorithm and with scaling = 0.05, using only one chain, and with a burn in stage of 200.

4. The resulting betas were again stored in empty matrices.

Considerations for the models:

Both implementations take into account model reliability and results. The precautions taken were introducing a burn in stage of 200 iterations in order to ensure that all of the posterior values come from a converged chain, trace plots that demonstrate that the chains actually converged, histograms to assess the distributions of the posterior values, and RMSE, MAE and R2 as diagnostic metrics.

Results:

The resulting trace plots show chains show convergence and favorable, though sometimes bimodal distributions. 

After evaluation, the best performing models are the manual implementation for in-store data and the PyMC implementation for the online data. In both cases, the in-store models show better results due to trends observed during the EDA. This was expected and still not unsustainable for the predictive abilities of the models.

The best performing models were exported as pkl files using joblib to allow replication.


- Frequentist models implementation.

See 'Frequentist Approach' file.

This stage uses statsmodels functions to create, train and evaluate the models.

The first section of the file is comprised by seaborn jointplots and correlation functions that indicate that the 2 most important variables - Quantity and Total Spent - have a very high correlations, with Pearson correlations of approximately 0.92-0.94, indicating a strong linear relationships.

Because of this, and Quantity being a countable feature, OLS and Poisson regressions were considered.

Even though it was considered, Poisson regression was discarded due to high equidispersion, or overdispersion (a high value of variance relative to the mean of the target variable). Note that Quantity was exponentiated in order to assess this and with the intention of performing the processes, since it was log-transformed during feature engineering (not done for OLS).


OLS models:

Both of these models (in-store and online) were implemented through the following steps:

1. Setting up the data like for Bayesian methods - creating the causal feature structure and separating the target value.

2. Using sm.OLS().fit() to create the models.

3. Printing a summary of the models focused on F-statistic and R2.

4. Validating models through the same metrics as before, RMSE, MAE and R2, and residual plots.

These processes resulted in very favorable metrics as expected due to the nature of the data (correlation), as well as low p-values associated to the F-statistic that indicate that the model is statistically significant and reliable. Finally, the residual plots show no significant trends and residual points centered around 0, varying from -0.4 to 0.5. All of these results point at effectivity and reliability.


These models were also stored as joblib objects for replication.


- Instructions and demonstration for usage.

In this section another dataset from Kaggle was harvested, and several steps are taken to achieve a structure that can be used to train, evaluate or use the models. This file is not intended to show that the model is adaptative, since the structure and scaling of the data is different to the training data. More specifically, the relationship between Total Spent (the most significant value for these models) and Quantity (the target value), is evidently different. This new data experiences higher spending per unit of stock needed, resulting on the model under estimating the amount of stock required.

Steps:

1. Loading joblib objects
2. Feature engineering the data to replicate the train dataset structure and characteristics.
3. Executing models and evaluating RMSE, MAE and R2.
4. Only for the Bayesian models, new data frames were created in order to capture the individual rows in which a 95% credible interval was able to accurate when predicting the Quantity needed.

- Limitations, future improvements and general suggestions.

1. This project was built on top of dirty, but artificial data. In an organic setting, factors like irregular inventory cycles, shifts in demand and production delays would be challenges to approach, but these could still make the models more robust for future usage.

2. Future versions of this project could benefit of the use of API's for deployment, or batch pipeline using tools.

3. A larger, richer and more diverse dataset could make this project more robust, and therefore able to predict stock needs for a general public as well as being more scalable.

4. This project would pair exceptionally well with a revenue predicting tool, taking into account that Total Spent was the most significant variable for stock optimization.

5. In future versions, it would be interesting to explore various implementations of the MCMC. Algorithms like NUTS might be able to perform differently and advantageously depending on the situation and needs.


License: This project is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) due to the use of external dataset(s).
