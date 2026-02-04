# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

# Create the DataFrame from the CSV file, parsing the date columns

rental = pd.read_csv('rental_info.csv', parse_dates=['rental_date', 'return_date'])
rental.head()
rental.info()

# Create a 'rental_length_days' column to count how many days a move is rented for

rental['rental_length_days'] = rental['return_date'] - rental['rental_date']
rental['rental_length_days'] = rental['rental_length_days'].dt.days

# Create two new columns for specific special features

rental['deleted_scenes'] = rental['special_features'].str.contains('Deleted Scenes')
rental['behind_the_scenes'] = rental['special_features'].str.contains('Behind the Scenes')

# Prepare X and y by selecting the columns to be used as features and target

X = rental.drop(["rental_length_days", "rental_date", "return_date", "special_features"], axis=1)
y = rental["rental_length_days"]

# Create a train test split with 80% split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

# Testing different Regression Models to find the one with the best MSE for the company (< 3), as requested.

# Lasso regression is a form of regularization for linear regression models. Regularization is a statistical method to reduce errors caused by overfitting on training data. 

lasso = Lasso(alpha=0.2, random_state=9)

lasso.fit(X_train, y_train)
lasso_coef = lasso.coef_

# Select the features with coefficient above 0

X_lasso_train, X_lasso_test = X_train.iloc[:, lasso_coef > 0], X_test.iloc[:, lasso_coef > 0]

# Instantiate a Linear Regression model, fit it on the lasso train data and predict on the lasso test data

linear = LinearRegression()
linear.fit(X_lasso_train, y_train)
linear_pred = linear.predict(X_lasso_test)

# Calculate mean squared error

linear_mse = mean_squared_error(y_test, linear_pred)
linear_mse
# Result is: 4.812297241276244

# Create hyperparameters for DecisionTreeRegressor

params = {"max_depth": np.arange(1,20,1),
         "min_samples_leaf": np.arange(1,11,1)}

# Instantiate the model

decisiontree = DecisionTreeRegressor()

# Use RandomizedSearchCV to go through the parameters and find the best ones

search_decisiontree = RandomizedSearchCV(decisiontree, params, cv=5, random_state=9)

search_decisiontree.fit(X_train, y_train)

decisiontree_params = search_decisiontree.best_params_

# Create a DecisionTreeRegressor model that uses the best parameters

dt_best = DecisionTreeRegressor(max_depth=decisiontree_params["max_depth"],
                                min_samples_leaf=decisiontree_params["min_samples_leaf"],
                                random_state=9)

# Fit on train data and predict on test data

dt_best.fit(X_train, y_train)
decision_pred = dt_best.predict(X_test)

# Calculate the mean squared error

mse_dt = mean_squared_error(y_test, decision_pred)
mse_dt
#Result is 2.26041606525774

# Create hyperparameters for RandomForestRegressor

params = {"max_depth": np.arange(1,20,1),
         "min_samples_leaf": np.arange(1,11, 1),
         "n_estimators": np.arange(1, 50, 1)}

# Instantiate the model

randomforest = RandomForestRegressor()

# Use RandomizedSearchCV to go through the parameters and find the best ones

search_randomforest = RandomizedSearchCV(randomforest, params, cv=5, random_state=9)

search_randomforest.fit(X_train, y_train)

randomforest_params = search_randomforest.best_params_

# Create a RandomForestRegressor model that uses the best parameters

rf_best = RandomForestRegressor(max_depth=randomforest_params["max_depth"],
                                min_samples_leaf=randomforest_params["min_samples_leaf"],
                                n_estimators=randomforest_params['n_estimators'],
                                random_state=9)

# Fit on train data and predict on test data

rf_best.fit(X_train, y_train)
rf_pred = rf_best.predict(X_test)

# Calculate the mean squared error

mse_rf = mean_squared_error(y_test, rf_pred)
mse_rf
#Result is 2.1113019519865777

# Create variables that have the best model and lowest MSE value

best_model = rf_best
best_mse = mse_rf
