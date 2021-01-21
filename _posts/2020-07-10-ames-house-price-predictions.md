---
title: "Regression Analysis: Predicting Ames Housing Market Prices"
date: 2020-07-10
tags: [machine learning, data science, regression]
header:
  image: "/images/projects/regression-melbourne-housing/melbourne.jpg"
excerpt: "Supervised learning models predicted housing prices by utilizing multiple housing price indicators."
---

<br>

Housing prices have steadily increased over the course of the past three decades with the exception of severe economic downturns such as the economic recession of 2008.
The housing market is not only a very strong economic indicator but it has a financial impact on anyone looking to own a home themselves.
To better understand the effects that individual factors have on the housing prices, I am interested in using supervised learning techniques to model housing prices.
By using machine learning techniques to do this the process can be automated to include a large amount of data points and different trends can be detected that may not be readily apparent to humans.


In this study, several types of supervised learning classification models were used to predict housing prices in Ames, Iowa. Models focused on utilizing multiple housing price indicators, including factors related to the size and location of the living spaces. The different models were compared to better understand their ability to utilize the data to accurately predict the housing market using multiple forms of statistical evaluation. The process used to undertake this study is as follows:

<br>

Initiation and Data Preprocessing
* Importing Packages and Files
* Defining Reusable Functions
* Data Exploration and Cleaning

Exploratory Data Analysis
* Analyzing the Distribution of the Target Variable
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


## Exploratory Data Analysis

### Analyzing the Distribution of the Target Variable


![png](output_17_0.png)


The mean sale price is \\$180,921 and the median sale price is \\$163,000. The distribution of the sale prices is skewed to the right. A logarithmic transformation can be used to make the sale prices more normally distributed prior to modeling.

### Identifying Statistically Significant Features




![png](output_20_1.png)


The above plot displays the ten continuous features with the highest linear relationship to the sales price. The units used to describe this is the absolute value of the correlation coefficient (range 0 to 1). Variables with a correlation coefficient of .5 or higher have a strong linear relationship with the sales price (variables with lower correlation coefficients are not shown here).

### Univariate Data Analysis


![png](output_23_0.png)


The above histograms display the distribution of the top features. The histograms are ordered based on the features' correlation to the sale price (most correlated to least correlated). As the correlation decreases, the distribution of the features have less of a resemblance to the distribution of the sale price.

### Bivariate Data Analysis


![png](output_26_0.png)


The above scatterplots display the relationship of the top features to the sale price. The scatterplots are ordered based on the features' correlation to the sale price (most correlated to least correlated). As the correlation decreases, features display less of a linear relationship with sales price.


### Multivariate Analysis of Key Features


![png](output_31_0.png)


The above scatterplots display the paired relationship of the top features to the sale price. The scatterplots are ordered based on the features' correlation to the sale price (most correlated to least correlated).




### Correlation of Variables


![png](output_36_0.png)


There are strong correlations among features that measure a similar quality of the homes (such as the year the house was built and year the garage was built).

### Descriptive Statistics and Boxplots




    Descriptive Statistics of Top Features



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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>OverallQual</th>
      <td>1094.0</td>
      <td>6.247715</td>
      <td>1.366797</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>7.00</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>GrLivArea</th>
      <td>1094.0</td>
      <td>1535.027422</td>
      <td>526.124028</td>
      <td>438.0</td>
      <td>1164.0</td>
      <td>1480.0</td>
      <td>1779.00</td>
      <td>5642.0</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>1094.0</td>
      <td>1.879342</td>
      <td>0.658586</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>GarageArea</th>
      <td>1094.0</td>
      <td>503.760512</td>
      <td>192.261314</td>
      <td>160.0</td>
      <td>360.0</td>
      <td>484.0</td>
      <td>602.50</td>
      <td>1418.0</td>
    </tr>
    <tr>
      <th>TotalBsmtSF</th>
      <td>1094.0</td>
      <td>1099.561243</td>
      <td>415.851262</td>
      <td>105.0</td>
      <td>816.0</td>
      <td>1023.0</td>
      <td>1345.50</td>
      <td>6110.0</td>
    </tr>
    <tr>
      <th>1stFlrSF</th>
      <td>1094.0</td>
      <td>1173.809872</td>
      <td>387.677463</td>
      <td>438.0</td>
      <td>894.0</td>
      <td>1097.0</td>
      <td>1413.50</td>
      <td>4692.0</td>
    </tr>
    <tr>
      <th>FullBath</th>
      <td>1094.0</td>
      <td>1.577697</td>
      <td>0.550219</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>TotRmsAbvGrd</th>
      <td>1094.0</td>
      <td>6.570384</td>
      <td>1.584486</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>7.00</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>YearBuilt</th>
      <td>1094.0</td>
      <td>1972.412249</td>
      <td>31.189752</td>
      <td>1880.0</td>
      <td>1953.0</td>
      <td>1975.0</td>
      <td>2003.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>YearRemodAdd</th>
      <td>1094.0</td>
      <td>1985.915905</td>
      <td>20.930772</td>
      <td>1950.0</td>
      <td>1967.0</td>
      <td>1995.0</td>
      <td>2005.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>1094.0</td>
      <td>1978.565814</td>
      <td>25.934444</td>
      <td>1900.0</td>
      <td>1960.0</td>
      <td>1982.0</td>
      <td>2003.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>1094.0</td>
      <td>109.855576</td>
      <td>190.667459</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>171.75</td>
      <td>1600.0</td>
    </tr>
    <tr>
      <th>SalePrice</th>
      <td>1094.0</td>
      <td>187033.263254</td>
      <td>83165.332151</td>
      <td>35311.0</td>
      <td>132500.0</td>
      <td>165750.0</td>
      <td>221000.00</td>
      <td>755000.0</td>
    </tr>
  </tbody>
</table>
</div>


Due to the prescence of outliers, the median (the colum denoted '50%') displays information that is more representative of the data.




![png](output_41_0.png)


These boxplots, along with all of the other visualizations, can be used to uncover outliers prior to modeling. Notable outliers can be observed in the features associated with the spatial dimensions of the homes.

## Predictive Modeling and Evaluation

Models are evaluated by using the following metrics on the validation set: R-squared value, root mean square error, and mean absolute error. Additionally, the residuals from the validation set are plotted and analyzed.


### Linear Regression (Predictive)



    Validation Set Evaluation

    R squared score:
    0.8937420137868703

    RMSE: 24990.84774368596

    MAE: 17603.94990048896




### K Nearest Neighbors



    Validation Set Evaluation

    R squared score:
    0.8461546705188671

    RMSE: 30070.618178354798

    MAE: 19883.733789954338

    CPU times: user 1.08 s, sys: 279 ms, total: 1.35 s
    Wall time: 408 ms




### Random Forest



    Validation Set Evaluation

    R squared score:
    0.8929342828451408

    RMSE: 25085.653158110843

    MAE: 16939.827465753424




## Creating Price Predictions For Unsold Homes


![png](output_69_0.png)


The best performing model (the predictive linear regression model) was used to predict the sale prices of unsold homes.

## Final Analysis and Conclusion

Understanding how to better utilize supervised modeling techniques to predict housing prices will give insight into which factors have the most effect on the prices of homes. Information about how such trends change over time can also be gained, which will be useful in understanding the real estate market which is a major economic indicator.

This study established the best suprvised modeling technique for predicting housing prices. The next step in using this data to gather insights from sales of homes would be to collect housing data from greater time spans (involving similar homes) and use them to train a model that will focus on seasonality and change over time. By being able to understand how such supervised learning models can be improved with the added context of time, housing prices can be predicted even more accurately and more information can be gained about the housing market that can provide actionable insights.
