---
title: "Regression Analysis: Predicting Melbourne Housing Market Prices"
date: 2019-05-29
tags: [machine learning, data science, regression]
header:
  image: "/images/projects/regression-melbourne-housing/melbourne.jpg"
excerpt: "Supervised learning models predicted housing prices by utilizing multiple housing price indicators."
---

The full code can be found [here](https://colab.research.google.com/drive/1gnWyBSlIkoKCc9R_VujD-VlE9mMzLdE9).

Housing prices have steadily increased over the course of the past three decades with the exception of severe economic downturns such as the economic recession of 2008.
The housing market is not only a very strong economic indicator but it has a financial impact on anyone looking to own a home themselves.
To better understand the effects that individual factors have on the housing prices, I am interested in using supervised learning techniques to model housing prices.
By using machine learning techniques to do this the process can be automated to include a large amount of data points and different trends can be detected that may not be readily apparent to humans.


In this study, several types of supervised learning classification models were used to predict housing prices in Melbourne, Australia. Models focused on utilizing multiple housing price indicators, including factors related to the size and location of the living spaces. The different models were compared to better understand their ability to utilize the data to accurately predict the housing market using multiple forms of statistical evaluation. The process used to undertake this study is as follows:



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





![png](https://raw.githubusercontent.com/donmacfoy/donmacfoy.github.io/master/images/projects/regression-melbourne-housing/output_12_0.png)



With the exception of the top selling areas, there is a similar amount of homes sold among the Post Codes and Suburbs with the most homes sold. There apprars to be a sharper disparity in the council areas with the homes sold which is more reflective of greater difference in size of these areas. Each of these variables result in a large number of classes with fewer datapoints in each, which may not be as useful for modeling.




![png](https://raw.githubusercontent.com/donmacfoy/donmacfoy.github.io/master/images/projects/regression-melbourne-housing/output_14_1.png)


Aside from a few top selling real estate agents, there's a large number of agents (over 400) that sold less than two thousand homes.



The vast majority of the transactions involved the homes being sold outright. Property being passed on or sold via auction make a minority of the transactions.


Most of the homes included in this data were houses, which outnumber units and townhouses combined.




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



### Gradient Boost



    accuracy score:
    0.7589473319543428

    cross validation:
    [0.74373468 0.73823348 0.76311434 0.76003719 0.74564779]

    RMSE: 0.214063296281816


The gradient boost model had the best performance out of all of the models run with all of the dataset's useful features. Cross vadidation showed few signs of overfitting with this model. The strength of this model when it comes to making predictions using this data comes from its ability to reduce error over multiple iterations, resulting in higher accuracy scores after a high number of iterations. Using all of the available features in the dataset could leave a model vulnerable to weak features (which could hurt accuracy), but the model overcomes that through its ability to directly focus reducing error. The random forest and knn models also had similar accuracy scores to the gradient boost model, but their cross validation scores were lower.


## Modeling the Data using PCA Components


### Gradient Boost


    accuracy score:
    0.7407961774045213

    cross validation:
    [0.72648411 0.70852419 0.71188499 0.72554533 0.71467114]

    RMSE: 0.22249309151772126




The gradient boost model had the best performance out of all of the models run with 20 of the dataset's PCA components. While cross vadidation did showed few signs of overfitting with this model, the scores were significantly lower than the accuracy score. This is a noticable difference from the gradient boost model that used all of the available features. Using a limited number of PCA components from the dataset likely removed some variance that was important to the predictive accuracy of the models. This can be seen in the sharp drop of performance in the naive bayes model. The random forest and knn models also had similar accuracy scores to the gradient boost model, but their cross validation scores were low as well.


## Modeling the Data using Features Chosen through the SelectKbest Function


### Gradient Boost


    accuracy score:
    0.7430477466561921

    cross validation:
    [0.73820231 0.72724553 0.72525356 0.74116231 0.73649944]

    RMSE: 0.22152464274783065


The gradient boost model had the best performance out of all of the models run with features chosen using the selectkbest function. Cross vadidation showed some signs of overfitting with this model. The cross validation scores with this model were slightly lower than the accuracy score, but the disparity wasn't as great as it was in the gradient boost model that used the PCA components. The gradient boost model using selectkbest still had lower accuracy than the  model that used all of the available features in the dataset, implying that the majority of the dataset's available features had useful variance when it came to the gradient boost. There was no general trend when in model performance when it comes to comparing full featured models and models that used sectkbest; Some models performed better using full features, and others performed better using selectkbest.


## Analysis and Conclusion

The gradient forest using all available features was by far the best model when it came to using information about homes sold to predict housing prices in Melbourne. It is also important to note that random forest and  KNN were the next best performing models. What held those two models back were their cross validation scores. KNN's relative success implies that the clustering of specific classes and ranges in this data have strong predictive value. The success of the full featured models in general show that most of the features used in the model have useful variance.

Understanding how to better utilize supervised modeling techniques to predict housing prices will give insight as to which factors have the most effect on the prices of homes. Information about how such trends change over time can also be gained, which will be useful in understanding the real estate market which is a major economic indicator.

This study established the best suprvised modeling technique for predicting housing prices. The next step in using this data to gather insights from sales of homes would be to collect housing data from greater time spans (involving similar homes) and use them to train a model that will account for seasonality and change over time. By being able to understand how such supervised learning models can be improved with the added context of time, housing prices can be predicted even more accurately and more information can be gained about the housing market that can provide actionable insights.
