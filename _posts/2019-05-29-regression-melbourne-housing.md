---
title: "Regression Analysis: Predicting Melbourne Housing Market Prices"
date: 2019-05-29
tags: [machine learning, data science, regression]
header:
  image: "/images/projects/regression-melbourne-housing/melbourne.jpg"
excerpt: "Supervised learning models predicted housing prices by utilizing multiple housing price indicators."
---

The full code can be found [here](https://github.com/donmacfoy/Portfolio/blob/master/Regression%20Analysis%20-%20Predicting%20Melbourne%20Housing%20Market%20Prices.ipynb).

Housing prices have steadily increased over the course of the past three decades with the exception of severe economic downturns such as the economic recession of 2008.
The housing market is not only a very strong economic indicator but it has a financial impact on anyone looking to own a home themselves.
To better understand the effects that individual factors have on the housing prices, I am interested in using supervised learning techniques to model housing prices.
By using machine learning techniques to do this the process can be automated to include a large amount of data points and different trends can be detected that may not be readily apparent to humans.


In this study, several types of supervised learning classification models were used to predict housing prices in Melbourne, Australia. Models focused on utilizing multiple housing price indicators, including factors related to the size and location of the living spaces. The different models were compared to better understand their ability to utilize the data to accurately predict the housing market using multiple forms of statistical evaluation. The process used to undertake this study is as follows:

<br>


Data Exploration and Analysis
* Viewing the Distribution of the Datapoints
* Checking the Range of the Classes
* Interpreting Descriptive Statistics

Preparing The Data For Modeling
* Imputing Outliers
* Normalization Linear Transformations
* Feature Selection

Modeling the Data
1. Using All Available Features
2. Using PCA Components
3. Using Selectkbest Function



## Data Exploration and Analysis

Dataset used for this study includes information about home purchases in Melbourne pertaining to the homes and the circumstances in which they were sold. Such information includes: the location of the homes, the number of rooms in the homes, the type of home, the method in which the home was sold, and the seller of the home.




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Address</th>
      <th>Bathroom</th>
      <th>BuildingArea</th>
      <th>CouncilArea</th>
      <th>Date</th>
      <th>Distance</th>
      <th>Method</th>
      <th>Postcode</th>
      <th>Price</th>
      <th>Propertycount</th>
      <th>Regionname</th>
      <th>Rooms</th>
      <th>SellerG</th>
      <th>Suburb</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49 Lithgow St</td>
      <td>1.493</td>
      <td>140.406</td>
      <td>Yarra</td>
      <td>736420</td>
      <td>3.000</td>
      <td>S</td>
      <td>3067</td>
      <td>1490000.000</td>
      <td>4019.000</td>
      <td>Northern Metropolitan</td>
      <td>3</td>
      <td>Jellis</td>
      <td>Abbotsford</td>
      <td>h</td>
    </tr>
    <tr>
      <th>1</th>
      <td>59A Turner St</td>
      <td>1.493</td>
      <td>140.406</td>
      <td>Yarra</td>
      <td>736420</td>
      <td>3.000</td>
      <td>S</td>
      <td>3067</td>
      <td>1220000.000</td>
      <td>4019.000</td>
      <td>Northern Metropolitan</td>
      <td>3</td>
      <td>Marshall</td>
      <td>Abbotsford</td>
      <td>h</td>
    </tr>
    <tr>
      <th>2</th>
      <td>119B Yarra St</td>
      <td>1.493</td>
      <td>140.406</td>
      <td>Yarra</td>
      <td>736420</td>
      <td>3.000</td>
      <td>S</td>
      <td>3067</td>
      <td>1420000.000</td>
      <td>4019.000</td>
      <td>Northern Metropolitan</td>
      <td>3</td>
      <td>Nelson</td>
      <td>Abbotsford</td>
      <td>h</td>
    </tr>
    <tr>
      <th>3</th>
      <td>68 Vida St</td>
      <td>1.493</td>
      <td>140.406</td>
      <td>Moonee Valley</td>
      <td>736420</td>
      <td>7.500</td>
      <td>S</td>
      <td>3040</td>
      <td>1515000.000</td>
      <td>1543.000</td>
      <td>Western Metropolitan</td>
      <td>3</td>
      <td>Barry</td>
      <td>Aberfeldie</td>
      <td>h</td>
    </tr>
    <tr>
      <th>4</th>
      <td>92 Clydesdale Rd</td>
      <td>1.141</td>
      <td>93.104</td>
      <td>Moonee Valley</td>
      <td>736420</td>
      <td>10.400</td>
      <td>S</td>
      <td>3042</td>
      <td>670000.000</td>
      <td>3464.000</td>
      <td>Western Metropolitan</td>
      <td>2</td>
      <td>Nelson</td>
      <td>Airport West</td>
      <td>h</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4/32 Earl St</td>
      <td>1.141</td>
      <td>93.104</td>
      <td>Moonee Valley</td>
      <td>736420</td>
      <td>10.400</td>
      <td>S</td>
      <td>3042</td>
      <td>530000.000</td>
      <td>3464.000</td>
      <td>Western Metropolitan</td>
      <td>2</td>
      <td>Jellis</td>
      <td>Airport West</td>
      <td>t</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3/74 Hawker St</td>
      <td>1.141</td>
      <td>93.104</td>
      <td>Moonee Valley</td>
      <td>736420</td>
      <td>10.400</td>
      <td>S</td>
      <td>3042</td>
      <td>540000.000</td>
      <td>3464.000</td>
      <td>Western Metropolitan</td>
      <td>2</td>
      <td>Barry</td>
      <td>Airport West</td>
      <td>u</td>
    </tr>
  </tbody>
</table>
</div>




![png](https://raw.githubusercontent.com/donmacfoy/donmacfoy.github.io/master/images/projects/regression-melbourne-housing/output_12_0.png)



With the exception of the top selling areas, there is a similar amount of homes sold among the Post Codes and Suburbs with the most homes sold. There apprars to be a sharper disparity in the council areas with the homes sold which is more reflective of greater difference in size of these areas. Each of these variables result in a large number of classes with fewer datapoints in each, which may not be as useful for modeling.




![png](https://raw.githubusercontent.com/donmacfoy/donmacfoy.github.io/master/images/projects/regression-melbourne-housing/output_14_1.png)


Aside from a few top selling real estate agents, there's a large number of agents (over 400) that sold less than two thousand homes.


```python
df.Method.value_counts()
```




    S     53789
    PI    14635
    SP    14005
    VB     9060
    SN     3989
    PN      959
    W       657
    SA      642
    SS      109
    Name: Method, dtype: int64



The vast majority of the transactions involved the homes being sold outright. Property being passed on or sold via auction make a minority of the transactions.


```python
df.Type.value_counts()
```




    h    69000
    u    18952
    t     9893
    Name: Type, dtype: int64



Most of the homes included in this data were houses, which outnumber units and townhouses combined.


```python
df.Regionname.value_counts()
```




    Southern Metropolitan         29385
    Northern Metropolitan         26329
    Western Metropolitan          18517
    Eastern Metropolitan          14766
    South-Eastern Metropolitan     6946
    Eastern Victoria                792
    Northern Victoria               759
    Western Victoria                351
    Name: Regionname, dtype: int64



By using the region name, the data falls into fewer, larger classes than with post codes, council areas or suburbs.




![png](https://raw.githubusercontent.com/donmacfoy/donmacfoy.github.io/master/images/projects/regression-melbourne-housing/output_22_0.png)


The frequency of property counts show has a distribution where there are thousands of observations for most ranges and there is little prescence of outliers. These qualities make property count likely to be a useful feature when interpreting the data.

The price variable, the outcome that I predicted on the other hand, showed signs of existing outliers. These outliers would be further explored and corrected prior to modeling.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bathroom</th>
      <th>BuildingArea</th>
      <th>Date</th>
      <th>Distance</th>
      <th>Price</th>
      <th>Propertycount</th>
      <th>Rooms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>97845.000</td>
      <td>97845.000</td>
      <td>97845.000</td>
      <td>97845.000</td>
      <td>97845.000</td>
      <td>97845.000</td>
      <td>97845.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.615</td>
      <td>155.333</td>
      <td>736519.522</td>
      <td>12.151</td>
      <td>1027307.202</td>
      <td>7602.077</td>
      <td>3.080</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.536</td>
      <td>161.045</td>
      <td>242.807</td>
      <td>7.351</td>
      <td>554515.464</td>
      <td>4426.052</td>
      <td>0.949</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>735991.000</td>
      <td>0.000</td>
      <td>85000.000</td>
      <td>39.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.141</td>
      <td>93.104</td>
      <td>736295.000</td>
      <td>6.700</td>
      <td>675000.000</td>
      <td>4380.000</td>
      <td>2.000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.493</td>
      <td>140.406</td>
      <td>736553.000</td>
      <td>11.000</td>
      <td>940000.000</td>
      <td>6786.000</td>
      <td>3.000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000</td>
      <td>210.576</td>
      <td>736700.000</td>
      <td>16.000</td>
      <td>1236714.827</td>
      <td>10412.000</td>
      <td>4.000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>12.000</td>
      <td>44515.000</td>
      <td>736980.000</td>
      <td>64.100</td>
      <td>11200000.000</td>
      <td>21650.000</td>
      <td>8.000</td>
    </tr>
  </tbody>
</table>
</div>



Extreme values could be found in the building area and property count variables. Data denoting the number of rooms and bathrooms on the other hand has much smaller ranges.

## Preparing The Data For Modeling


Categorical variables that didn't have a very large number of classes were kept for the model to prevent its performance from being hampered.




```python
%%time

## Train Test Split Original Variables And K Selected Variables for Modeling

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

kx_train, kx_test, ky_train, ky_test = train_test_split(k_predictors, y, test_size=0.2, random_state=21)

px_train, px_test, py_train, py_test = train_test_split(pca_components, y, test_size=0.2, random_state=21)

```


## Modeling the Data using all available Features

### Linear Regression


```python
%%time

## train and fit model

regr = linear_model.LinearRegression().fit(x_train, y_train)
```

    CPU times: user 365 ms, sys: 37.7 ms, total: 402 ms
    Wall time: 154 ms



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(regr.score(x_test, y_test))+'\n')

print("cross validation:\n" + str(cross_val_score(regr, x_test, y_test, cv=5))+'\n')

print('RMSE: ' + str(rmse(regr.predict(x_test), y_test)))

```

    accuracy score:
    0.6554818379673324

    cross validation:
    [0.65020109 0.64819587 0.6672538  0.66042741 0.64593618]

    RMSE: 0.25591297063312535
    CPU times: user 933 ms, sys: 222 ms, total: 1.15 s
    Wall time: 184 ms


### K Nearest Neighbors


```python
%%time

## train and fit model

knn = neighbors.KNeighborsRegressor(n_neighbors=10).fit(x_train, y_train)
```

    CPU times: user 5.49 s, sys: 55.9 ms, total: 5.55 s
    Wall time: 5.6 s



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(knn.score(x_test, y_test))+'\n')

print("cross validation:\n" + str(cross_val_score(knn, x_test, y_test, cv=5))+'\n')

print('RMSE: ' + str(rmse(knn.predict(x_test), y_test)))

```

    accuracy score:
    0.7377193118403025

    cross validation:
    [0.68162125 0.66202621 0.69920477 0.69558934 0.6870499 ]

    RMSE: 0.22329006212615407
    CPU times: user 47.8 s, sys: 278 ms, total: 48.1 s
    Wall time: 48.5 s


### Decision Tree


```python
%%time

## train and fit model

decision_tree = tree.DecisionTreeRegressor(
    max_features=10,
    max_depth=100,
    ).fit(x_train, y_train)

```

    CPU times: user 140 ms, sys: 27.7 ms, total: 168 ms
    Wall time: 166 ms



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(decision_tree.score(x_test, y_test))+'\n')

print("cross validation:\n" + str(cross_val_score(decision_tree, x_test, y_test, cv=5))+'\n')

print('RMSE: ' + str(rmse(decision_tree.predict(x_test), y_test)))

```

    accuracy score:
    0.7479054604916736

    cross validation:
    [0.63769201 0.64628347 0.67084755 0.63400314 0.66111049]

    RMSE: 0.21891118741869403
    CPU times: user 209 ms, sys: 35 ms, total: 244 ms
    Wall time: 244 ms


### Random Forest


```python
%%time

## Fit and Train Model

rfr = ensemble.RandomForestRegressor(
    max_features=10,
    max_depth=100,
    ).fit(x_train, y_train)

```

    CPU times: user 829 ms, sys: 38.3 ms, total: 867 ms
    Wall time: 881 ms



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(rfr.score(x_test, y_test))+'\n')

print("cross validation:\n" + str(cross_val_score(rfr, x_test, y_test, cv=5))+'\n')

print('RMSE: ' + str(rmse(rfr.predict(x_test), y_test)))

```

    accuracy score:
    0.7587281677852276

    cross validation:
    [0.7001267  0.70259043 0.72497107 0.71053659 0.71227038]

    RMSE: 0.21416058694023726
    CPU times: user 1.08 s, sys: 65.2 ms, total: 1.14 s
    Wall time: 1.15 s


### Gradient Boost


```python
%%time

## train and fit model

re = ensemble.GradientBoostingRegressor()

parameters = {
              'n_estimators': list(np.arange(200, 401, 50)),
              'max_depth': list(range(1,5)),
             }

r2_scorer = make_scorer(r2_score)

reg = GridSearchCV(re, parameters, scoring=r2_scorer).fit(x_train,  y_train)

## Show Best Parameters
print(reg.best_params_)

```

    {'max_depth': 4, 'n_estimators': 400}
    CPU times: user 14min 42s, sys: 8.49 s, total: 14min 50s
    Wall time: 15min 3s



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(reg.score(x_test, y_test))+'\n')

print("cross validation:\n" + str(cross_val_score(reg, x_test, y_test, cv=5))+'\n')

print('RMSE: ' + str(rmse(reg.predict(x_test), y_test)))

```

    accuracy score:
    0.7589473319543428

    cross validation:
    [0.74373468 0.73823348 0.76311434 0.76003719 0.74564779]

    RMSE: 0.214063296281816
    CPU times: user 13min 46s, sys: 6.24 s, total: 13min 52s
    Wall time: 18min 56s


The gradient boost model had the best performance out of all of the models run with all of the dataset's useful features. Cross vadidation showed few signs of overfitting with this model. The strength of this model when it comes to making predictions using this data comes from its ability to reduce error over multiple iterations, resulting in higher accuracy scores after a high number of iterations. Using all of the available features in the dataset could leave a model vulnerable to weak features (which could hurt accuracy), but the model overcomes that through its ability to directly focus reducing error. The random forest and knn models also had similar accuracy scores to the gradient boost model, but their cross validation scores were lower.


## Modeling the Data using PCA Components

### Linear Regression


```python
%%time

## train and fit model

p_regr = linear_model.LinearRegression().fit(px_train, py_train)

```

    CPU times: user 79.3 ms, sys: 79.1 ms, total: 158 ms
    Wall time: 170 ms



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(p_regr.score(px_test, py_test))+'\n')

print("cross validation:\n" + str(cross_val_score(p_regr, px_test, py_test, cv=5))+'\n')

print('RMSE: ' + str(rmse(p_regr.predict(px_test), py_test)))

```

    accuracy score:
    0.5030878678329703

    cross validation:
    [0.50806286 0.49279613 0.49611531 0.51229806 0.50048328]

    RMSE: 0.30806022598109145
    CPU times: user 402 ms, sys: 137 ms, total: 540 ms
    Wall time: 89.9 ms


### K Nearest Neighbors


```python
%%time

## train and fit model

p_knn = neighbors.KNeighborsRegressor(n_neighbors=10).fit(px_train, py_train)
```

    CPU times: user 120 ms, sys: 24.6 ms, total: 145 ms
    Wall time: 157 ms



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(p_knn.score(px_test, py_test))+'\n')

print("cross validation:\n" + str(cross_val_score(p_knn, px_test, py_test, cv=5))+'\n')

print('RMSE: ' + str(rmse(p_knn.predict(px_test), py_test)))

```

    accuracy score:
    0.7325633961116327

    cross validation:
    [0.6825508  0.66356092 0.6704457  0.67694743 0.66932064]

    RMSE: 0.22599886281472986
    CPU times: user 2 s, sys: 36.9 ms, total: 2.04 s
    Wall time: 2.07 s


### Decision Tree


```python
%%time

## train and fit model

p_decision_tree = tree.DecisionTreeRegressor(
    max_features=10,
    max_depth=100,
    ).fit(px_train, py_train)

```

    CPU times: user 489 ms, sys: 6.56 ms, total: 496 ms
    Wall time: 499 ms



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(p_decision_tree.score(px_test, py_test))+'\n')

print("cross validation:\n" + str(cross_val_score(decision_tree, px_test, py_test, cv=5))+'\n')

print('RMSE: ' + str(rmse(p_decision_tree.predict(px_test), py_test)))

```

    accuracy score:
    0.7481527829844821

    cross validation:
    [0.66116194 0.62890079 0.6206433  0.61572942 0.63935557]

    RMSE: 0.21931301612750023
    CPU times: user 654 ms, sys: 6.78 ms, total: 661 ms
    Wall time: 664 ms


### Random Forest


```python
%%time

## Fit and Train Model

p_rfr = ensemble.RandomForestRegressor(
    max_features=10,
    max_depth=100,
    ).fit(px_train, py_train)

```

    CPU times: user 3.43 s, sys: 40.6 ms, total: 3.47 s
    Wall time: 3.52 s



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(p_rfr.score(px_test, py_test))+'\n')

print("cross validation:\n" + str(cross_val_score(p_rfr, px_test, py_test, cv=5))+'\n')

print('RMSE: ' + str(rmse(p_rfr.predict(px_test), py_test)))

```

    accuracy score:
    0.7557480495570719

    cross validation:
    [0.71295825 0.69382045 0.68699159 0.69955297 0.69345797]

    RMSE: 0.21598065293150928
    CPU times: user 4.4 s, sys: 54.1 ms, total: 4.46 s
    Wall time: 4.52 s


### Gradient Boost


```python
%%time

## train and fit model

p_re = ensemble.GradientBoostingRegressor()

parameters = {
              'n_estimators': list(np.arange(200, 401, 50)),
              'max_depth': list(range(1,5))
             }

acc_scorer = make_scorer(r2_score)

p_reg = GridSearchCV(p_re, parameters, scoring=acc_scorer).fit(px_train, py_train)

## Show Best Parameters
print(p_reg.best_params_)

```

    {'max_depth': 4, 'n_estimators': 400}
    CPU times: user 18min 21s, sys: 6.32 s, total: 18min 28s
    Wall time: 18min 40s



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(p_reg.score(px_test, py_test))+'\n')

print("cross validation:\n" + str(cross_val_score(p_reg, px_test, py_test, cv=5))+'\n')

print('RMSE: ' + str(rmse(p_reg.predict(px_test), py_test)))

```

    accuracy score:
    0.7407961774045213

    cross validation:
    [0.72648411 0.70852419 0.71188499 0.72554533 0.71467114]

    RMSE: 0.22249309151772126
    CPU times: user 18min 45s, sys: 8.5 s, total: 18min 53s
    Wall time: 19min 10s



The gradient boost model had the best performance out of all of the models run with 20 of the dataset's PCA components. While cross vadidation did showed few signs of overfitting with this model, the scores were significantly lower than the accuracy score. This is a noticable difference from the gradient boost model that used all of the available features. Using a limited number of PCA components from the dataset likely removed some variance that was important to the predictive accuracy of the models. This can be seen in the sharp drop of performance in the naive bayes model. The random forest and knn models also had similar accuracy scores to the gradient boost model, but their cross validation scores were low as well.



## Modeling the Data using Features Chosen through the SelectKbest Function

### Linear Regression


```python
%%time

## train and fit model

k_regr = linear_model.LinearRegression().fit(kx_train, ky_train)
```

    CPU times: user 106 ms, sys: 70.3 ms, total: 176 ms
    Wall time: 138 ms



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(k_regr.score(kx_test, ky_test))+'\n')

print("cross validation:\n" + str(cross_val_score(k_regr, kx_test, ky_test, cv=5))+'\n')

print('RMSE: ' + str(rmse(k_regr.predict(kx_test), ky_test)))

```

    accuracy score:
    0.6423702734460872

    cross validation:
    [0.65007419 0.63049271 0.62826453 0.65465068 0.64323639]

    RMSE: 0.261344060753883
    CPU times: user 393 ms, sys: 192 ms, total: 585 ms
    Wall time: 125 ms


### K Nearest Neighbors


```python
%%time

## train and fit model

k_knn = neighbors.KNeighborsRegressor(n_neighbors=10).fit(kx_train, ky_train)

```

    CPU times: user 3.95 s, sys: 49.2 ms, total: 3.99 s
    Wall time: 4.12 s



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(k_knn.score(kx_test, ky_test))+'\n')

print("cross validation:\n" + str(cross_val_score(k_knn, kx_test, ky_test, cv=5))+'\n')

print('RMSE: ' + str(rmse(k_knn.predict(kx_test), ky_test)))

```

    accuracy score:
    0.7304730679361253

    cross validation:
    [0.70513445 0.68780232 0.69425137 0.70833818 0.69695175]

    RMSE: 0.22688036568655084
    CPU times: user 18.2 s, sys: 151 ms, total: 18.3 s
    Wall time: 18.6 s


### Decision Tree


```python
%%time

## train and fit model

k_decision_tree = tree.DecisionTreeRegressor(
    max_features=10,
    max_depth=100,
    ).fit(kx_train, ky_train)

```

    CPU times: user 102 ms, sys: 7.92 ms, total: 110 ms
    Wall time: 111 ms



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(k_decision_tree.score(kx_test, ky_test))+'\n')

print("cross validation:\n" + str(cross_val_score(k_decision_tree, kx_test, ky_test, cv=5))+'\n')

print('RMSE: ' + str(rmse(k_decision_tree.predict(kx_test), ky_test)))

```

    accuracy score:
    0.7436031805267569

    cross validation:
    [0.68601962 0.67089887 0.66503723 0.68787319 0.67661231]

    RMSE: 0.22128508685200665
    CPU times: user 118 ms, sys: 7.66 ms, total: 125 ms
    Wall time: 129 ms


### Random Forest


```python
%%time

## Fit and Train Model

k_rfr = ensemble.RandomForestRegressor(
    max_features=10,
    max_depth=100,
    ).fit(kx_train, ky_train)

```

    CPU times: user 663 ms, sys: 19.2 ms, total: 682 ms
    Wall time: 693 ms



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(k_rfr.score(kx_test, ky_test))+'\n')

print("cross validation:\n" + str(cross_val_score(k_rfr, kx_test, ky_test, cv=5))+'\n')

print('RMSE: ' + str(rmse(k_rfr.predict(kx_test), ky_test)))

```

    accuracy score:
    0.7470835079617548

    cross validation:
    [0.70600971 0.69916833 0.69222898 0.7144088  0.7063068 ]

    RMSE: 0.21977809481277225
    CPU times: user 771 ms, sys: 31.5 ms, total: 803 ms
    Wall time: 811 ms


### Gradient Boost


```python
%%time

## train and fit model

k_re = ensemble.GradientBoostingRegressor()

parameters = {
              'n_estimators': list(np.arange(200, 401, 50)),
              'max_depth': list(range(1,5))
             }

acc_scorer = make_scorer(r2_score)

k_reg = GridSearchCV(k_re, parameters, scoring=acc_scorer).fit(kx_train, ky_train)

## Show Best Parameters
print(k_reg.best_params_)

```

    {'max_depth': 4, 'n_estimators': 400}
    CPU times: user 9min 44s, sys: 3.97 s, total: 9min 48s
    Wall time: 9min 54s



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(k_reg.score(kx_test, ky_test))+'\n')

print("cross validation:\n" + str(cross_val_score(k_reg, kx_test, ky_test, cv=5))+'\n')

print('RMSE: ' + str(rmse(k_reg.predict(kx_test), ky_test)))

```

    accuracy score:
    0.7430477466561921

    cross validation:
    [0.73820231 0.72724553 0.72525356 0.74116231 0.73649944]

    RMSE: 0.22152464274783065
    CPU times: user 8min 4s, sys: 3.71 s, total: 8min 8s
    Wall time: 8min 15s


The gradient boost model had the best performance out of all of the models run with features chosen using the selectkbest function. Cross vadidation showed some signs of overfitting with this model. The cross validation scores with this model were slightly lower than the accuracy score, but the disparity wasn't as great as it was in the gradient boost model that used the PCA components. The gradient boost model using selectkbest still had lower accuracy than the  model that used all of the available features in the dataset, implying that the majority of the dataset's available features had useful variance when it came to the gradient boost. There was no general trend when in model performance when it comes to comparing full featured models and models that used sectkbest; Some models performed better using full features, and others performed better using selectkbest.


## Analysis and Conclusion

The gradient forest using all available features was by far the best model when it came to using information about homes sold to predict housing prices in Melbourne. It is also important to note that random forest and  KNN were the next best performing models. What held those two models back were their cross validation scores. KNN's relative success implies that the clustering of specific classes and ranges in this data have strong predictive value. The success of the full featured models in general show that most of the features used in the model have useful variance.

Understanding how to better utilize supervised modeling techniques to predict housing prices will give insight as to which factors have the most effect on the prices of homes. Information about how such trends change over time can also be gained, which will be useful in understanding the real estate market which is a major economic indicator.

This study established the best suprvised modeling technique for predicting housing prices. The next step in using this data to gather insights from sales of homes would be to collect housing data from greater time spans (involving similar homes) and use them to train a model that will account for seasonality and change over time. By being able to understand how such supervised learning models can be improved with the added context of time, housing prices can be predicted even more accurately and more information can be gained about the housing market that can provide actionable insights.


```python

```
