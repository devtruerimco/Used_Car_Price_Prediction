# Used_Car_Price_Prediction
This project aims at predicting the price of a used car, using Sklearn's supervised machine learning techniques integrated with Sklearn library.It is obvious that this is a regression problem, and predictions are made using a dataset of used car sales in the American car market. Several regression techniques, such as linear regression, decision trees, and random forests of decision trees, have been investigated. Their performance was compared to see which one worked best with our dataset.

### TOOLS

Most of the project has been developed using Python as the programming language of choice and the following libraries:

* Scikit-Learn, regression models and cross validation techniques.
* Seaborn, visualization and plotting library
* Pandas, data analysis purposes.

### PROBLEM

The problem of used car price prediction has some value because various studies show that the used car market is destined for continuous growth in the short term.

Research shows that prices of used cars have gone up by 41% in USA market in 2022 since 2018-2019 pre-COVID era. This problem, however, is difficult to solve because the car's value is determined by a variety of factors such as year of registration, manufacturer, model, mileage, horsepower, origin, and a variety of other specific information such as type of fuel and braking system, condition of bodywork and interiors, interior materials (plastics and leather), safety index, type of change (manual, assisted, automatic, semi-automatic), number of doors, number of previous owners, if it was previously owned by. Unfortunately, only a small portion of this information is available, so it is critical to relate the accuracy results to the features available for analysis.


### DATASET  

Consists of the following columns initially:
1. ID
2. Price
3. Year
4. Mileage
5. City
6. State
7. Vin
8. Make
9. Model

### ANALYSIS

Each of the three prediction models used has a unique set of parameters (fit intercept, normalize, and copy X for the linear regressor, max depth for random forest and decision trees). The model's performance may improve or deteriorate depending on the values assumed by these parameters. Hyperparameter Tuning refers to the process of determining the optimal value for each parameter in relation to the training set.
This is a time-consuming and resource-intensive process. The entire project was built by utilizing the functionalities provided by Python's Sci-Kit Learn library (Sklearn), so the parameter tuning process is handled by a function called GridSearchCV: This method fits with every possible combination of parameter values specified in a grid of parameters while also running the Cross-Validation.

 * Linear regression:
  
  LinearRegression(copy_X=True, fit_intercept=False, n_jobs=1, normalize=True)
  
 * Decision Tree regressor:
 
  DecisionTreeRegressor(criterion='mse', max_depth=17, max_features=None,
           max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           presort=False, random_state=None, splitter='best')
           
 * Random Forest regressor:
 
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=18,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)
           
           
 ### CONCLUSIONS 
 
 We can see that in the cross validation analysis, linear regression and random forest are the ones that perform better with RMSE's lower than decision tree regressor.
 
 Mean scores returned by each model are as follows:
  
 * Linear Regression:  0.7779025106692288
 * Decision Tree Regressor: 0.7395719775661147
 * Random Forest Regressor: Mean: 0.7964076921587058
