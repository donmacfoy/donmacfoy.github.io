# Regression Analysis: Predicting Ames Housing Market Prices

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


## Initiation and Data Preprocessing

### Importing Packages and Files


```python
%%time

## import packages

import warnings

import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from feature_engine import outlier_removers as outr
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFE, RFECV, SequentialFeatureSelector
from sklearn.feature_selection import f_regression, SelectKBest, chi2, mutual_info_regression

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_rows', None)
pd.options.mode.chained_assignment = None

sns.set_style('darkgrid')
sns.color_palette('Paired')

# Suppress Warnings
warnings.filterwarnings(action="ignore")

%matplotlib inline
```

    CPU times: user 1.47 s, sys: 558 ms, total: 2.03 s
    Wall time: 3.1 s



```python
%%time

## import files

# read files
df_train_raw = pd.read_csv('train.csv')
df_test_raw = pd.read_csv('test.csv')

# saving multiple copies of training dataset
df_train = df_train_raw.copy()
df_train_plot = df_train_raw.copy()
```

    CPU times: user 37.9 ms, sys: 8.61 ms, total: 46.5 ms
    Wall time: 49.6 ms


### Defining Reusable Functions


```python
%%time

## defining reusable functions

# create a function to analyze columns

def feature_analysis(dataframe):
    
    # include a column for dtypes of each feature
    feature_analysis_df = pd.DataFrame(dataframe.dtypes)
    feature_analysis_df.columns = ['dtypes']
    
    # include a column for number of unique values of each feature
    feature_analysis_df['nunique'] = 0
    for feat in feature_analysis_df.index:
        feature_analysis_df.loc[feat,'nunique'] = len(dataframe.loc[:,feat].unique())
        
    # include a column for number of unique values of each feature    
    feature_analysis_df['isnull'] = 0
    for feat in feature_analysis_df.index:
        feature_analysis_df.loc[feat,'isnull'] = dataframe.loc[:,feat].isnull().sum(axis = 0)
        
    # print dataset characteristics
    print('\nDataset Characteristics \n')
    print('rows:', dataframe.shape[0])
    print('columns:', dataframe.shape[1])
    print(feature_analysis_df)
    print()    
    
# a function to calculate rmse for model evaluation

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

```

    CPU times: user 3 µs, sys: 0 ns, total: 3 µs
    Wall time: 5.96 µs



```python
%%time

## defining reusable pipeline components 

# drop features with large amounts of null values features

def drop_select_features(dataframe):
    return dataframe.drop(['Id', 'Alley', 'PoolQC', 'Fence', 'FireplaceQu', 'MiscFeature'],1)

drop_features = FunctionTransformer(drop_select_features)

# impute null values with string denoting missingness

def impute_null_categorical_values(dataframe):
    return dataframe.fillna('unknown')

impute_null_categorical = FunctionTransformer(impute_null_categorical_values)

# continuous features

def select_continuous_features(dataframe):
    return dataframe[list(dataframe.columns[dataframe.dtypes == 'int64'])]

continuous_features = FunctionTransformer(select_continuous_features)

# categorical features

def select_categorical_features(dataframe):
    return dataframe[list(dataframe.columns[dataframe.dtypes != 'int64'])]

categorical_features = FunctionTransformer(select_categorical_features)

# a custom transformer to pass unchanged features through a pipeline

def identity_transformer(dataframe):
    return dataframe

identity = FunctionTransformer(identity_transformer)

# a custom transformer to change a numpy matrix into a dataframe in a pipeline

def make_dataframe(dataframe):
    return pd.DataFrame(dataframe)

dataframe = FunctionTransformer(make_dataframe)

# create transformer function to add .0001 to all values

def make_non_zero(dataframe):
    return pd.DataFrame(dataframe).applymap(lambda x: x + .0001)

non_zero = FunctionTransformer(make_non_zero)

# create transformer function to one hot encode dataframes

def make_one_hot(dataframe):
    return pd.get_dummies(dataframe)

one_hot = FunctionTransformer(make_one_hot)

# create transformer to label one hot features

def label_categorical_features(dataframe):
    
    output = pd.DataFrame()
    
    for name in dataframe.columns:
        
        output[name] = dataframe[name].apply(lambda x: name + ' ' + str(x))
        
    return output

label_categorical = FunctionTransformer(label_categorical_features)
```

    CPU times: user 29 µs, sys: 1 µs, total: 30 µs
    Wall time: 31 µs


### Data Exploration and Cleaning

#### Checking for Correct Data Types


```python
%%time

## display dataset characteristics

#feature_analysis(df_train_raw)

## split raw dataset for field-specific data cleaning

# continuous variables
df_train_raw_continuous = df_train_raw[list(df_train_raw.columns[df_train_raw.dtypes != 'object'])
                                      ].drop('SalePrice',1)

# categorical variables
df_train_raw_categorical = df_train_raw[list(df_train_raw.columns[df_train_raw.dtypes == 'object'])]

# target variable
df_train_raw_target = df_train_raw['SalePrice']

print()
print('Continuous Features:')
display(df_train_raw_continuous.columns)
print()
print('Categorical Features:')
display(df_train_raw_categorical.columns)
print()
print('Target Variable:')
display(df_train_raw['SalePrice'].name)
print()


```

    
    Continuous Features:



    Index(['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
           'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
           'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
           'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
           'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
           'MiscVal', 'MoSold', 'YrSold'],
          dtype='object')


    
    Categorical Features:



    Index(['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
           'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
           'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
           'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
           'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
           'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
           'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
           'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
           'SaleType', 'SaleCondition'],
          dtype='object')


    
    Target Variable:



    'SalePrice'


    
    CPU times: user 17.7 ms, sys: 4.92 ms, total: 22.6 ms
    Wall time: 34.3 ms


The type of data contained within each feature is shown above. The target variable, sale price, contains continuous data.

#### Visualizing Missing Values



```python
%%time

## check for null values

#feature_analysis(df_train_raw_continuous)
#feature_analysis(df_train_raw_categorical)

## visualizing the prescence of null values

cmap=sns.color_palette("light:b_r")

fig, ax = plt.subplots(figsize=(16,8))         
ax = sns.heatmap(df_train_plot.isnull(), cbar=False, cmap=cmap, alpha=.5)
ax.set_title('Visualization of Null Values (By Feature)')
ax.set_yticks([])
plt.show()
```


![png](output_13_0.png)


    CPU times: user 4.01 s, sys: 415 ms, total: 4.42 s
    Wall time: 4.27 s


The white spaces above the feature names in the visualization represent represent areas where the data is missing. An overwhelming amount of null values were found among the following features: 'Alley', 'PoolQC', 'Fence', 'FireplaceQu', and 'MiscFeature'. These features will need to be dropped prior to modeling. Null values among the remaining features can be reliably imputed.

## Exploratory Data Analysis

### Analyzing the Distribution of the Target Variable


```python

## ploting the distribution of the target variable

plt.figure(figsize=(15,5))
ax = sns.histplot(x='SalePrice', data=df_train_plot)
ax.set_title('Distribution of Sale Prices')
ax.set_xlabel('Price (USD)')
ax.set_ylabel('Number of Homes')
plt.show()
```


![png](output_17_0.png)


The mean sale price is \\$180,921 and the median sale price is \\$163,000. The distribution of the sale prices is skewed to the right. A logarithmic transformation can be used to make the sale prices more normally distributed prior to modeling.

### Identifying Statistically Significant Features


```python

## analyzing continuous variables

# isolating continuous variables
df_train_plot = drop_select_features(df_train_raw).dropna()

df_train_continuous_plot = df_train_plot[list(df_train_plot.columns[df_train_plot.dtypes != 'object'])]

# identifying continuous variables with highest relationship to the target

# calculating p-values
continuous_pvalues = SelectKBest(score_func=f_regression, k='all').fit(
    df_train_continuous_plot.drop('SalePrice',1), df_train_continuous_plot['SalePrice']).pvalues_

# calculating correlation
continuous_correlation = abs(df_train_continuous_plot.corr()['SalePrice'])

# displaying relationship of features to target (sorted by ascending p-values)
print()

relationship_to_target = pd.DataFrame([df_train_continuous_plot.columns, 
                                       continuous_pvalues, continuous_correlation
                                      ]).T.rename({0:'feature name', 
                                                   1:'p-value', 
                                                   2:'correlation'}, axis=1
                                                 ).sort_values(by='p-value')

fig, ax = plt.subplots(1,1, figsize=(14,6))
ax = sns.barplot(x="feature name", 
                 y="correlation", 
                 data=relationship_to_target.iloc[0:10,:])
ax.set_title('Features with Highest Relationship to Target')
ax.set_xlabel('Feature Names')
ax.set_ylabel('Correlation to Sale Price')

plt.show()

```

    



![png](output_20_1.png)


The above plot displays the ten continuous features with the highest linear relationship to the sales price. The units used to describe this is the absolute value of the correlation coefficient (range 0 to 1). Variables with a correlation coefficient of .5 or higher have a strong linear relationship with the sales price (variables with lower correlation coefficients are not shown here).

### Univariate Data Analysis


```python

## visualizing the distribution of continuous variables

df_train_plot[relationship_to_target['feature name'][:12]
             ].hist(figsize=(16,8), 
                    bins=50, 
                    xlabelsize=8, 
                    ylabelsize=8, 
                    layout=(3,4))
plt.show()
```


![png](output_23_0.png)


The above histograms display the distribution of the top features. The histograms are ordered based on the features' correlation to the sale price (most correlated to least correlated). As the correlation decreases, the distribution of the features have less of a resemblance to the distribution of the sale price.

### Bivariate Data Analysis


```python

## visualizing the features' relationship to the target variable

# defining data to plot
df_to_plot = df_train_plot[relationship_to_target['feature name'][:12]]
target_to_plot = df_train_plot['SalePrice']

# defining plot dimensions
num_cols = 4
hspace = 0.35
wspace = 0.25
figsize = (16,12)

# plotting data
num_rows = len(df_to_plot.columns) // num_cols + 1

fig = plt.figure(figsize=figsize)
fig.subplots_adjust(hspace=hspace, wspace=wspace)

for i in range(len(df_to_plot.columns) + 1)[1:]:
    plt.subplot(num_rows, num_cols, i)
    plt.scatter(df_train_plot[df_to_plot.columns[i-1]],
                target_to_plot)
    plt.title(df_to_plot.columns[i-1])
```


![png](output_26_0.png)


The above scatterplots display the relationship of the top features to the sale price. The scatterplots are ordered based on the features' correlation to the sale price (most correlated to least correlated). As the correlation decreases, features display less of a linear relationship with sales price.


```python

## potting the distribution of the overall quality and target variables

fig, ax = plt.subplots(1,2, figsize=(14,10))

# potting the distribution of the above grade living area of the homes
plt.subplot(2, 2, 1)
ax[0] = sns.histplot(x='GrLivArea', data=df_train_plot)
ax[0].set_title('Distribution of the Above Grade Living Area')
ax[0].set_xlabel('Above Grade Living Area (sq. ft)')
ax[0].set_ylabel('Number of Homes')

# potting the distribution of the target variable in relation to overall quality
plt.subplot(2, 2, 2)
ax[1] = sns.regplot(data=df_train_plot, x="GrLivArea", y="SalePrice",line_kws={'color':'grey'} )
ax[1].set_title('Above Grade Living Area v. Sale Price')
ax[1].set_xlabel('Above Grade Living Area (sq. ft)')
ax[1].set_ylabel('Sale Price (USD)')

# potting the distribution of the overall quality of the homes
plt.subplot(2, 2, 3)
ax[0] = sns.countplot(x='OverallQual', data=df_train_plot)
ax[0].set_title('Distribution of the Overall Quality')
ax[0].set_xlabel('Quality Rating')
ax[0].set_ylabel('Number of Homes')

# potting the distribution of the target variable in relation to overall quality
plt.subplot(2, 2, 4)
ax[1] = sns.boxplot(data=df_train_plot, x="OverallQual", y="SalePrice" )
ax[1].set_title('Overall Quality v. Sale Price')
ax[1].set_xlabel('Overall Quality Rating')
ax[1].set_ylabel('Sale Price')

plt.show()
```


![png](output_28_0.png)


There are a couple of outliers with abnormally high above grade living area but low sale prices (see top right scatter plot).

### Multivariate Analysis of Key Features


```python

## plot multivariate relationships

sns.pairplot(df_train_plot[['OverallQual', 'GrLivArea', 'GarageCars', 
                            'TotalBsmtSF', 'FullBath', 
                            'SalePrice']], 
             diag_kind = 'kde', 
             hue = 'SalePrice', 
             plot_kws = {'alpha': 0.6},
             diag_kws = {'alpha': 0.6})

plt.show()
```


![png](output_31_0.png)


The above scatterplots display the paired relationship of the top features to the sale price. The scatterplots are ordered based on the features' correlation to the sale price (most correlated to least correlated). 


```python

## potting the distribution of sales prices in relation to the above ground living area ang overall quality rating

palette = sns.color_palette('flare', as_cmap=True)

fig, ax = plt.subplots(1,1, figsize=(14,6))

ax = sns.scatterplot(
     data=df_train_plot.rename(columns={"SalePrice": "Sale Price"}),
     x="OverallQual", y="GrLivArea", hue='Sale Price', palette=palette )
ax.set_title('Overall Quality Rating v. Above Grade Living Area v. Sale Price')
ax.set_xlabel('Overall Quality Rating')
ax.set_ylabel('Above Grade Living Area (sq. ft)')

plt.show()
```


![png](output_33_0.png)


With the exception of a couple of outliers, quality rating and above grade living area when paired together have a strong linear relationship with sale price.

### Correlation of Variables


```python

## Visualizing the Correlatedness of the Session variables

fig, ax = plt.subplots(figsize=(12,8))         
sns.heatmap(df_train[relationship_to_target['feature name']].corr(), cmap='RdBu_r', center=0)
plt.show()
```


![png](output_36_0.png)


There are strong correlations among features that measure a similar quality of the homes (such as the year the house was built and year the garage was built).

### Descriptive Statistics and Boxplots


```python

## descriptive statistics of most significant features
print()
print('Descriptive Statistics of Top Features')

display(df_train_plot[relationship_to_target.sort_values(by='p-value'
                                                )['feature name'
                                                 ][:12].append(pd.Series('SalePrice'))].describe().T)
```

    
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


```python

## visualizing the features' relationship to the target variable

# defining data to plot
df_to_plot = df_train_plot[relationship_to_target['feature name'][:12]]
target_to_plot = df_train_plot['SalePrice']

# defining plot dimensions
num_cols = 4
hspace = 0.35
wspace = 0.25
figsize = (16,12)

# plotting data
num_rows = len(df_to_plot.columns) // num_cols + 1

fig = plt.figure(figsize=figsize)
fig.subplots_adjust(hspace=hspace, wspace=wspace)

for i in range(len(df_to_plot.columns) + 1)[1:]:
    plt.subplot(num_rows, num_cols, i)
    plt.boxplot(df_train_plot[df_to_plot.columns[i-1]])
    plt.xticks([1], [''])
    plt.title(df_to_plot.columns[i-1])
```


![png](output_41_0.png)


These boxplots, along with all of the other visualizations, can be used to uncover outliers prior to modeling. Notable outliers can be observed in the features associated with the spatial dimensions of the homes. 

## Predictive Modeling and Evaluation

Models are evaluated by using the following metrics on the validation set: R-squared value, root mean square error, and mean absolute error. Additionally, the residuals from the validation set are plotted and analyzed.

### Data Preprocessing


```python
%%time

# drop extreme outliers

df_train = df_train.loc[~((df_train['GrLivArea'] > 4000) & ( df_train['SalePrice'] < 200000))]

## Establish Feature and Outcome Variables to be Used for Modeling Based on Original Features

x = df_train.drop('SalePrice',1)
y = df_train['SalePrice']

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=25)

x_train = x_train.reset_index(drop=True)
x_val = x_val.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)
```

    CPU times: user 7.42 ms, sys: 2.8 ms, total: 10.2 ms
    Wall time: 11 ms



```python
%%time

## data preprocessing

# feature names
feature_names_categorical = x_train[list(x_train.columns[x_train.dtypes == 'object'])].columns
feature_names_continuous = x_train[list(x_train.columns[x_train.dtypes != 'object'])].columns

# categorical pipeline
pipe_categorical = Pipeline([('categorical', categorical_features), 
                             ('impute_categorical', impute_null_categorical), 
                             ('label_categorical', label_categorical),
                             ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

# continuous pipeline
pipe_continuous = Pipeline([('impute_continuous', SimpleImputer(strategy='median')),
                            ('standardscaler', StandardScaler())])

# combine feature pipelines
all_features = ColumnTransformer(transformers=[('categorical_features', 
                                                pipe_categorical, 
                                                feature_names_categorical),
                                               ('continuous_features', 
                                                pipe_continuous, 
                                                feature_names_continuous) ], remainder='drop')
```

    CPU times: user 4.27 ms, sys: 266 µs, total: 4.53 ms
    Wall time: 4.42 ms


### Linear Regression (Predictive)

#### Creating and Evaluating the Model 


```python
%%time

## train and fit model

# modeling pipeline
pipe = Pipeline([('all_features', all_features), 
                 ('selectkbest', SelectKBest(score_func=f_regression)),
                 ('model', Ridge())]).fit(x_train, y_train)

# searching for best parameters
pipe = GridSearchCV(estimator=pipe, 
                    param_grid = {'selectkbest__k': [60, 70, 80, 90, 100], 
                                  'selectkbest__score_func': [chi2, f_regression, mutual_info_regression]}, 
                    n_jobs=-1
                   ).fit(x_train, y_train)

print()
print('Best Gridsearch Parameters')
print()
print(pipe.best_params_)

pipe = pipe.best_estimator_
```

    
    Best Gridsearch Parameters
    
    {'selectkbest__k': 70, 'selectkbest__score_func': <function mutual_info_regression at 0x1a187554d0>}
    CPU times: user 5.47 s, sys: 213 ms, total: 5.68 s
    Wall time: 33.8 s



```python
%%time

## Model Evaluation 

print('Train Set Evaluation')
print()
print("R squared score:\n" + str(pipe.score(x_train, y_train)))
print()
print('RMSE: ' + str(rmse(pipe.predict(x_train), y_train)))
print()
print('MAE: ' + str(mean_absolute_error(y_train, pipe.predict(x_train))))
print()
print("cross validation:\n" + str(cross_val_score(pipe, 
                                                  x_train, 
                                                  y_train, 
                                                  cv=5)))
print()
```

    Train Set Evaluation
    
    R squared score:
    0.9078017815150131
    
    RMSE: 24485.98872987244
    
    MAE: 16222.578178154079
    
    cross validation:
    [0.88701416 0.89346184 0.86075577 0.86421796 0.88893552]
    
    CPU times: user 15 s, sys: 1.11 s, total: 16.1 s
    Wall time: 13.4 s



```python
%%time

## Model Evaluation

print('Validation Set Evaluation')
print()
print("R squared score:\n" + str(pipe.score(x_val, y_val)))
print()
print('RMSE: ' + str(rmse(pipe.predict(x_val), y_val)))
print()
print('MAE: ' + str(mean_absolute_error(y_val, pipe.predict(x_val))))
print()

```

    Validation Set Evaluation
    
    R squared score:
    0.8937420137868703
    
    RMSE: 24990.84774368596
    
    MAE: 17603.94990048896
    
    CPU times: user 832 ms, sys: 274 ms, total: 1.11 s
    Wall time: 530 ms



```python
%%time

## get coefficients 

# get feature names
one_hot_encoded_features = pipe.named_steps['all_features'
                ].named_transformers_['categorical_features'
                                     ].named_steps['one_hot'].get_feature_names()

all_feature_names = list(one_hot_encoded_features) + list(feature_names_continuous)

selected_feature_names = np.array(all_feature_names)[pipe.named_steps['selectkbest'].get_support()]

# get coefficients
linear_model_coefficients = pipe.named_steps['model'].coef_

# display coefficients
coefficient_analysis = pd.DataFrame([selected_feature_names, 
              linear_model_coefficients, 
              abs(linear_model_coefficients)]
            ).T.sort_values(2, 
                            ascending=False
                           ).rename(columns={0:"features", 
                                             1:"coefficients", 
                                             2:"absolute value coefficients"})

display(coefficient_analysis.head(10))
```


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
      <th>features</th>
      <th>coefficients</th>
      <th>absolute value coefficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>44</th>
      <td>x41_SaleType New</td>
      <td>34326.2</td>
      <td>34326.2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>x8_Neighborhood NoRidge</td>
      <td>31584.5</td>
      <td>31584.5</td>
    </tr>
    <tr>
      <th>13</th>
      <td>x18_ExterQual Ex</td>
      <td>23793.7</td>
      <td>23793.7</td>
    </tr>
    <tr>
      <th>27</th>
      <td>x30_KitchenQual Ex</td>
      <td>18543.6</td>
      <td>18543.6</td>
    </tr>
    <tr>
      <th>58</th>
      <td>2ndFlrSF</td>
      <td>17577.1</td>
      <td>17577.1</td>
    </tr>
    <tr>
      <th>45</th>
      <td>x42_SaleCondition Partial</td>
      <td>-16113.5</td>
      <td>16113.5</td>
    </tr>
    <tr>
      <th>59</th>
      <td>GrLivArea</td>
      <td>15850.5</td>
      <td>15850.5</td>
    </tr>
    <tr>
      <th>49</th>
      <td>OverallQual</td>
      <td>14860.2</td>
      <td>14860.2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>x21_BsmtQual Gd</td>
      <td>-12789.3</td>
      <td>12789.3</td>
    </tr>
    <tr>
      <th>56</th>
      <td>TotalBsmtSF</td>
      <td>12528</td>
      <td>12528</td>
    </tr>
  </tbody>
</table>
</div>


    CPU times: user 67.2 ms, sys: 34.1 ms, total: 101 ms
    Wall time: 40.5 ms



```python
%%time

## potting the distribution of predicted and actual values

to_plot = pd.DataFrame()

to_plot['Actual'] = y_val
to_plot['Predicted'] = pipe.predict(x_val)
to_plot['Predicted - Actual'] = to_plot['Predicted'] - to_plot['Actual']


fig, ax = plt.subplots(1,2, figsize=(14,12))

# potting the predicted and actual values
plt.subplot(2, 1, 1)
ax[0] = sns.scatterplot(
     data=to_plot,
     x="Actual", y="Predicted")
ax[0].set_title('Actual Sale v. Predicted Sale Price')
ax[0].set_xlabel('Actual Sale Price')
ax[0].set_ylabel('Predicted Sale Price')

# potting the difference between the predicted and actual values
plt.subplot(2, 1, 2)
ax[1] = sns.scatterplot(
     data=to_plot,
     x="Actual", y='Predicted - Actual')
ax[1].set_title('Actual Sale v. Model Error')
ax[1].set_xlabel('Actual Sale Price')
ax[1].set_ylabel('Difference from Sale Price')
ax[1].axhline(0, ls='--')
plt.show()
```


![png](output_54_0.png)


    CPU times: user 1.44 s, sys: 198 ms, total: 1.64 s
    Wall time: 1.27 s


### K Nearest Neighbors

#### Creating and Evaluating the Model 


```python
%%time

## train and fit model

# modeling pipeline
pipe = Pipeline([('all_features', all_features), 
                 ('selectkbest', SelectKBest(score_func=f_regression)),
                 ('model', KNeighborsRegressor())]).fit(x_train, y_train)

# searching for best parameters
pipe = GridSearchCV(estimator=pipe, 
                    param_grid = {'selectkbest__k': [60, 70, 80, 90, 100], 
                                  'selectkbest__score_func': [chi2, f_regression, mutual_info_regression]}, 
                    n_jobs=-1
                   ).fit(x_train, y_train)

print()
print('Best Gridsearch Parameters')
print()
print(pipe.best_params_)

pipe = pipe.best_estimator_
```

    
    Best Gridsearch Parameters
    
    {'selectkbest__k': 70, 'selectkbest__score_func': <function f_regression at 0x1a1873c050>}
    CPU times: user 2.19 s, sys: 145 ms, total: 2.34 s
    Wall time: 31 s



```python
%%time

## Model Evaluation 

print('Train Set Evaluation')
print()
print("R squared score:\n" + str(pipe.score(x_train, y_train)))
print()
print('RMSE: ' + str(rmse(pipe.predict(x_train), y_train)))
print()
print('MAE: ' + str(mean_absolute_error(y_train, pipe.predict(x_train))))
print()
print("cross validation:\n" + str(cross_val_score(pipe, 
                                                  x_train, 
                                                  y_train, 
                                                  cv=5)))
print()
```

    Train Set Evaluation
    
    R squared score:
    0.8917388669235061
    
    RMSE: 26533.38536368112
    
    MAE: 16371.09274509804
    
    cross validation:
    [0.8512213  0.84271515 0.8218693  0.78663132 0.87980621]
    
    CPU times: user 4.53 s, sys: 940 ms, total: 5.47 s
    Wall time: 1.78 s



```python
%%time

## Model Evaluation

print('Validation Set Evaluation')
print()
print("R squared score:\n" + str(pipe.score(x_val, y_val)))
print()
print('RMSE: ' + str(rmse(pipe.predict(x_val), y_val)))
print()
print('MAE: ' + str(mean_absolute_error(y_val, pipe.predict(x_val))))
print()

```

    Validation Set Evaluation
    
    R squared score:
    0.8461546705188671
    
    RMSE: 30070.618178354798
    
    MAE: 19883.733789954338
    
    CPU times: user 1.08 s, sys: 279 ms, total: 1.35 s
    Wall time: 408 ms



```python
%%time

## potting the distribution of predicted and actual values

to_plot = pd.DataFrame()

to_plot['Actual'] = y_val
to_plot['Predicted'] = pipe.predict(x_val)
to_plot['Predicted - Actual'] = to_plot['Predicted'] - to_plot['Actual']


fig, ax = plt.subplots(1,2, figsize=(14,12))

# potting the predicted and actual values
plt.subplot(2, 1, 1)
ax[0] = sns.scatterplot(
     data=to_plot,
     x="Actual", y="Predicted")
ax[0].set_title('Actual Sale v. Predicted Sale Price')
ax[0].set_xlabel('Actual Sale Price')
ax[0].set_ylabel('Predicted Sale Price')

# potting the difference between the predicted and actual values
plt.subplot(2, 1, 2)
ax[1] = sns.scatterplot(
     data=to_plot,
     x="Actual", y='Predicted - Actual')
ax[1].set_title('Actual Sale v. Model Error')
ax[1].set_xlabel('Actual Sale Price')
ax[1].set_ylabel('Difference from Sale Price')
ax[1].axhline(0, ls='--')
plt.show()
```


![png](output_60_0.png)


    CPU times: user 1.12 s, sys: 157 ms, total: 1.28 s
    Wall time: 728 ms


### Random Forest

#### Creating and Evaluating the Model 


```python
%%time

## train and fit model

# modeling pipeline
pipe = Pipeline([('all_features', all_features), 
                 ('selectkbest', SelectKBest(score_func=f_regression)),
                 ('model', RandomForestRegressor())]).fit(x_train, y_train)

# searching for best parameters
pipe = GridSearchCV(estimator=pipe, 
                    param_grid = {'selectkbest__k': [60, 70, 80, 90, 100], 
                                  'selectkbest__score_func': [chi2, f_regression, mutual_info_regression]}, 
                    n_jobs=-1
                   ).fit(x_train, y_train)

print()
print('Best Gridsearch Parameters')
print()
print(pipe.best_params_)

pipe = pipe.best_estimator_
```

    
    Best Gridsearch Parameters
    
    {'selectkbest__k': 60, 'selectkbest__score_func': <function mutual_info_regression at 0x1a187554d0>}
    CPU times: user 7.01 s, sys: 260 ms, total: 7.27 s
    Wall time: 55.5 s



```python
%%time

## Model Evaluation 

print('Train Set Evaluation')
print()
print("R squared score:\n" + str(pipe.score(x_train, y_train)))
print()
print('RMSE: ' + str(rmse(pipe.predict(x_train), y_train)))
print()
print('MAE: ' + str(mean_absolute_error(y_train, pipe.predict(x_train))))
print()
print("cross validation:\n" + str(cross_val_score(pipe, 
                                                  x_train, 
                                                  y_train, 
                                                  cv=5)))
print()
```

    Train Set Evaluation
    
    R squared score:
    0.9830064239569182
    
    RMSE: 10512.324783863847
    
    MAE: 6567.703421568628
    
    cross validation:
    [0.89921584 0.85327783 0.89249043 0.81669124 0.89419588]
    
    CPU times: user 14.9 s, sys: 288 ms, total: 15.2 s
    Wall time: 15.6 s



```python
%%time

## Model Evaluation

print('Validation Set Evaluation')
print()
print("R squared score:\n" + str(pipe.score(x_val, y_val)))
print()
print('RMSE: ' + str(rmse(pipe.predict(x_val), y_val)))
print()
print('MAE: ' + str(mean_absolute_error(y_val, pipe.predict(x_val))))
print()

```

    Validation Set Evaluation
    
    R squared score:
    0.8929342828451408
    
    RMSE: 25085.653158110843
    
    MAE: 16939.827465753424
    
    CPU times: user 302 ms, sys: 14.9 ms, total: 317 ms
    Wall time: 321 ms



```python
%%time

## potting the distribution of predicted and actual values

to_plot = pd.DataFrame()

to_plot['Actual'] = y_val
to_plot['Predicted'] = pipe.predict(x_val)
to_plot['Predicted - Actual'] = to_plot['Predicted'] - to_plot['Actual']


fig, ax = plt.subplots(1,2, figsize=(14,12))

# potting the predicted and actual values
plt.subplot(2, 1, 1)
ax[0] = sns.scatterplot(
     data=to_plot,
     x="Actual", y="Predicted")
ax[0].set_title('Actual Sale v. Predicted Sale Price')
ax[0].set_xlabel('Actual Sale Price')
ax[0].set_ylabel('Predicted Sale Price')

# potting the difference between the predicted and actual values
plt.subplot(2, 1, 2)
ax[1] = sns.scatterplot(
     data=to_plot,
     x="Actual", y='Predicted - Actual')
ax[1].set_title('Actual Sale v. Model Error')
ax[1].set_xlabel('Actual Sale Price')
ax[1].set_ylabel('Difference from Sale Price')
ax[1].axhline(0, ls='--')
plt.show()
```


![png](output_66_0.png)


    CPU times: user 800 ms, sys: 85.7 ms, total: 885 ms
    Wall time: 666 ms


## Creating Price Predictions For Unsold Homes


```python
%%time

## train and fit model

# modeling pipeline
pipe = Pipeline([('all_features', all_features), 
                 ('selectkbest', SelectKBest(score_func=f_regression)),
                 ('model', Ridge())]).fit(x_train, y_train)

# searching for best parameters
pipe = GridSearchCV(estimator=pipe, 
                    param_grid = {'selectkbest__k': [60, 70, 80, 90, 100], 
                                  'selectkbest__score_func': [chi2, f_regression, mutual_info_regression]}, 
                    n_jobs=-1
                   ).fit(x_train, y_train)

print()
print('Best Gridsearch Parameters')
print()
print(pipe.best_params_)

pipe = pipe.best_estimator_
```

    
    Best Gridsearch Parameters
    
    {'selectkbest__k': 90, 'selectkbest__score_func': <function mutual_info_regression at 0x1a187554d0>}
    CPU times: user 3.84 s, sys: 166 ms, total: 4.01 s
    Wall time: 22.7 s



```python
## potting the distribution of the overall quality and target variables

fig, ax = plt.subplots(figsize=(12,8))  
# potting the distribution of the above grade living area of the homes

ax = sns.histplot(x=pipe.predict(df_test_raw))
ax.set_title('Distribution of Predicted House Sale prices')
ax.set_xlabel('Above Grade Living Area (sq. ft)')
ax.set_ylabel('Number of Homes')
plt.show()
```


![png](output_69_0.png)


The best performing model (the predictive linear regression model) was used to predict the sale prices of unsold homes.

## Final Analysis and Conclusion

Understanding how to better utilize supervised modeling techniques to predict housing prices will give insight into which factors have the most effect on the prices of homes. Information about how such trends change over time can also be gained, which will be useful in understanding the real estate market which is a major economic indicator. 

This study established the best suprvised modeling technique for predicting housing prices. The next step in using this data to gather insights from sales of homes would be to collect housing data from greater time spans (involving similar homes) and use them to train a model that will focus on seasonality and change over time. By being able to understand how such supervised learning models can be improved with the added context of time, housing prices can be predicted even more accurately and more information can be gained about the housing market that can provide actionable insights.


```python

```
