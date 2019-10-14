---
title: "Sentiment Analysis: Classifying Amazon Healthcare Product Reviews"
date: 2019-05-22
tags: [machine learning, data science, classification, natural language processing]
header:
  image: "/images/projects/sentiment-analysis-amazon-products/amazon.jpg"
excerpt: "Sentiment analysis was conducted to quantitatively judge subjective amazon reviews as positive or negative."
---

The full code can be found [here](https://github.com/donmacfoy/Portfolio/blob/master/sentiment-analysis-amazon-products.ipynb).




Sentiment analysis is a technique that uses machine learning and mathematical models to quantitatively analyze subjective information such as text. Due to an exponentially increasing amount of data being generated, companies rely on machine learning techniques such as sentiment analysis to gather insights from this data that can help them make impactful decisions. Sentiment analysis has many applications in product management such as customer sentiment analysis. By being able to understand information gained from customer feedback, decisionmakers can efficiently create strategies to improve product performance. To better understand how product feedback can be utilized, I am interested in using sentiment analysis to analyze customer reviews on healthcare products sold on Amazon.

In this study, sentiment analysis was done on healthcare product reviews using several types of supervised learning classification models. Models focused exclusively on utilizing the review text data to better gauge the impact of the sentiment analysis techniques. The different models were compared to better understand their ability to analyze the product review data. The process used to undertake this study is as follows:

Data Exploration and Analysis
* Analyzing the Sources of the Text Information
* Understanding the Sentiment of the Reviews Based on the Rating
* Viewing the Distribution of the Overall Rating

Preparing The Data For Modeling
* Labeling the Reviews Based on Rating
* Vectorizing the Text Data
* Dealing With Class Imbalance
* Feature Selection

Modeling the Data
* Naive Bayes
* K Nearest Neighbors
* Decision Trees
* Random Forest
* Logistic Regression (and Lasso and Ridge)
* Support Vector Classifier
* Gradient Boost




## Data Exploration and Analysis

Dataset used for this study includes product reviews and information about the context of the reviews.
Such information includes: the time the reviews were posted, the product ID, a helpfulness rating, a review summary, and an overall rating of the products.
The data being used in this study comes from the Amazon website and reflect data was collected between 1996 and 2014.


There were reviews for 18534 different products included in this study. This means that there was an average of about 19 reviews per product.


There were reviews from 38609 different reviewers included in this study. This means that there was an average of about 9 reviews per reviewer.

A review of a user who gave the product a 5 rating:


    'This is very nice. You pull out on the magnifier when you want the light to come on, then slide it back in. I would recommend buying this if you need something with a light that you can easily put in your pocket or purse.'



A review of a user who gave the product a 4 rating:



    "What I liked was the quality of the lens and the built in light.  Then lens had no discernable distortion anywhere.  It magnified everything evenly without the ripples and  distortion that I've seen with other low cost magnifiers.  This light is a nice touch and easy to use.  If you want it on just pull the lens out a bit.  It is focused very close to the center of what you will be look at and provides nice, even coverage.What I didn't like was the brightness (actually dimmness) of the light and where it is focused.  LEDs can be lots brighter, I know as I've seen them.  Also, the light focuses at the center of you field of view but only when the lens is too close to be focused properly.Bottom line is this is a good value for a magnifier and could have been made great with better quality control.BTW, I feel that honest, effective reviews can take the place of first-hand experiences that are lacking in online shopping. I've always appreciated the help I've received from other reviewers and work hard to return the favor as best as I can.  I hope you found this review helpful and if there was anything you thought was lacking or unclear leave a comment and I'll do what I can to fix it."



A review of a user who gave the product a 3 rating:




    'This magnifier has nothing to cover it when not in use.I compared it with a Carson 3x magnifier and this one did not seem as clear as the Carson - hard to see the furigana clearly in Japanese comic books with this one.The Carson ones come with a cover.I would recommend the Carson 5x if you are looking for good size enlarging.Carson MiniBrite 5x Power Slide- Out MagnifierI wish I had skipped the 3x and gone with the 5x only.For this one in the 3x, the light works well and lights things up nicely.Carson MiniBrite 5x Power Slide- Out Magnifier'



A review of a user who gave the product a 2 rating:




    'Bought for my mother due her eye sight going downhill. She said she still can not read what she wants to she said her little magnifier that has a light on it is better. She said she would not recommend.'



A review of a user who gave the product a 1 rating:




    'ONE STAR:The Maxell LR44 10-pack photo shows the new hologram packaging, but I received the old orange & black packaging.The batteries are stale. Lights powered by them are semi-bright, and only last a day or so.The orange & black pack rates 1-star.FIVE STARS:From the same supplier, MyBatterySupplier, I ordered the50-pack, which did come in the new hologram package, and the difference was dramatic.  Lights powered by the batteries were brilliant, and I expect them to last much longer.The new hologram pack rates 5-stars.'



There seems to be a large shift in sentiment from positive to negative as product ratings change from 4 to 3.


```python
df.overall.value_counts()
```




    5    211633
    4     68168
    3     33254
    2     16754
    1     16546
    Name: overall, dtype: int64



The distribution of reviews skew greatly toward a 5 rating. Since reviews will be labeled based on the ratings, class balancing will need to be done to reduce the impact of the label that will reflect the 5 ratings.

## Preparing The Data For Modeling

To prepare the data for modeling, the review text data was isolated and a new feature was engineered engineered to label the reviews as positive or negative. The review data was vectorized and the number of features was reduced using SVD and the selectKbest function.


Reviews accompanied by a rating of 4 or higher were labeled as positive (1) and reviews of 3 and below were labeled as negative or (0).


The positive reviews make a majority of the reviews while the negative reviews make the minority. Class balancing was done so that the models wouldn't indiscriminately predict the dominant class. The majority class was downsampled and the minority class was upsampled.


```python
%%time

## Train Test Split Original Variables And K Selected Variables for Modeling

x_train, x_test, y_train, y_test = train_test_split(k_predictors, y, test_size=0.2, random_state=20)
```

The x and y variables represent the variables to be used for training and testing the different supervised learning models.

## Modeling the Data

### Naive Bayes


```python
%%time

## train and fit model

bnb = BernoulliNB().fit(x_train, y_train)


```

    CPU times: user 328 ms, sys: 56.2 ms, total: 384 ms
    Wall time: 285 ms



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(bnb.score(x_test, y_test))+'\n')

print("cross validation:\n" + str(cross_val_score(bnb, x_test, y_test, cv=5))+'\n')

print("cross validation with AUC:\n" + str(cross_val_score(bnb, x_test, y_test, cv=5, scoring='roc_auc'))+'\n')

print("confusion matrix:\n" + str(confusion_matrix(y_test, bnb.predict(x_test)))+'\n')

print(classification_report(y_test, bnb.predict(x_test)))


```

    accuracy score:
    0.66785

    cross validation:
    [0.66975    0.6705     0.67233333 0.66341667 0.67483333]

    cross validation with AUC:
    [0.73179936 0.7283914  0.73166425 0.72167188 0.7349367 ]

    confusion matrix:
    [[19852  9953]
     [ 9976 20219]]

                  precision    recall  f1-score   support

               0       0.67      0.67      0.67     29805
               1       0.67      0.67      0.67     30195

       micro avg       0.67      0.67      0.67     60000
       macro avg       0.67      0.67      0.67     60000
    weighted avg       0.67      0.67      0.67     60000

    CPU times: user 3.71 s, sys: 628 ms, total: 4.33 s
    Wall time: 1.15 s


Naive bayes had relatively low accuracy compared to most of the other models.
The cross validation showed that overfitting was not greatly present with this model.
The model was slightly better at predicting the positive class.
The model assumes that the variables are uncorrelated, which is true because they have been reduced to svd components. However this also means that the model's performance may have been negatively impacted by its inability to capture the combined effect of multiple variables on the outcome. For example a word like 'good' may have a positive connotation, but another word like 'not' could change the context of 'good' to have a negative meaning ('not good'). Naive bayes would fail to capture the combined meaning of the two words.



### Decision Tree





```python
%%time

## train and fit model

decision_tree = tree.DecisionTreeClassifier(
    criterion='entropy',
    max_features=6,
    max_depth=25,
    ).fit(x_train, y_train)


```

    CPU times: user 4.09 s, sys: 33.6 ms, total: 4.12 s
    Wall time: 4.12 s



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(decision_tree.score(x_test, y_test))+'\n')

print("cross validation:\n" + str(cross_val_score(decision_tree, x_test, y_test, cv=5))+'\n')

print("cross validation with AUC:\n" + str(cross_val_score(decision_tree, x_test, y_test, cv=5, scoring='roc_auc'))+'\n')

print("confusion matrix:\n" + str(confusion_matrix(y_test, decision_tree.predict(x_test)))+'\n')

print(classification_report(y_test, decision_tree.predict(x_test)))

```

    accuracy score:
    0.8057333333333333

    cross validation:
    [0.648      0.6555     0.65475    0.65008333 0.64966667]

    cross validation with AUC:
    [0.65708832 0.65760371 0.65757921 0.66166587 0.65967743]

    confusion matrix:
    [[26571  3234]
     [ 8422 21773]]

                  precision    recall  f1-score   support

               0       0.76      0.89      0.82     29805
               1       0.87      0.72      0.79     30195

       micro avg       0.81      0.81      0.81     60000
       macro avg       0.81      0.81      0.80     60000
    weighted avg       0.82      0.81      0.80     60000

    CPU times: user 7.56 s, sys: 119 ms, total: 7.68 s
    Wall time: 7.68 s


The decision tree had high accuracy compared to the other model types in this study.
The cross validation showed that overfitting was not greatly present with this model.
The model was better at predicting the negative class.
The model's reliance on binary divisions likely improved its ability to capture nuance within the text and allows for greater discernment between reviews with correlated words and phrases.


### Random Forest




```python
%%time

## Fit and Train Model

rfc = ensemble.RandomForestClassifier(
    criterion='entropy',
    max_features=15,
    max_depth=100,
    ).fit(x_train, y_train)


```

    CPU times: user 1min 1s, sys: 125 ms, total: 1min 1s
    Wall time: 1min 1s



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(rfc.score(x_test, y_test))+'\n')

print("cross validation:\n" + str(cross_val_score(rfc, x_test, y_test, cv=5))+'\n')

print("cross validation with AUC:\n" + str(cross_val_score(rfc, x_test, y_test, cv=5, scoring='roc_auc'))+'\n')

print("confusion matrix:\n" + str(confusion_matrix(y_test, rfc.predict(x_test)))+'\n')

print(classification_report(y_test, rfc.predict(x_test)))

```

    accuracy score:
    0.8415

    cross validation:
    [0.70925    0.706      0.70208333 0.69708333 0.70225   ]

    cross validation with AUC:
    [0.79241176 0.79025421 0.79212652 0.78454937 0.79060043]

    confusion matrix:
    [[27548  2257]
     [ 7253 22942]]

                  precision    recall  f1-score   support

               0       0.79      0.92      0.85     29805
               1       0.91      0.76      0.83     30195

       micro avg       0.84      0.84      0.84     60000
       macro avg       0.85      0.84      0.84     60000
    weighted avg       0.85      0.84      0.84     60000

    CPU times: user 1min 46s, sys: 651 ms, total: 1min 47s
    Wall time: 1min 48s


Random forest had relatively high accuracy compared to most of the other models.
The cross validation showed that overfitting had very little prescence with this model.
The model was better at predicting the negative outcome class.
The model's success likely comes from it not having to rely on each set of data being evaluated only once.
By being able to base its evaluations on multiple sub decision trees classifications are only finalized after multiple iterations.
By building on the ability of the decision tree to divide the data based on varying contextual information, the random forest model was able to sustain relatively high performance.


### Logistic Regression



```python
%%time

## train and fit model

lr = LogisticRegression(fit_intercept=False).fit(x_train, y_train)

```

    CPU times: user 1.8 s, sys: 135 ms, total: 1.94 s
    Wall time: 1.98 s



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(lr.score(x_test, y_test))+'\n')

print("cross validation:\n" + str(cross_val_score(lr, x_test, y_test, cv=5))+'\n')

print("cross validation with AUC:\n" + str(cross_val_score(lr, x_test, y_test, cv=5, scoring='roc_auc'))+'\n')

print("confusion matrix:\n" + str(confusion_matrix(y_test, lr.predict(x_test)))+'\n')

print(classification_report(y_test, lr.predict(x_test)))

```

    accuracy score:
    0.7146

    cross validation:
    [0.71716667 0.71066667 0.71458333 0.71016667 0.71908333]

    cross validation with AUC:
    [0.79167359 0.78817354 0.79008466 0.78647806 0.79518103]

    confusion matrix:
    [[21379  8426]
     [ 8698 21497]]

                  precision    recall  f1-score   support

               0       0.71      0.72      0.71     29805
               1       0.72      0.71      0.72     30195

       micro avg       0.71      0.71      0.71     60000
       macro avg       0.71      0.71      0.71     60000
    weighted avg       0.71      0.71      0.71     60000

    CPU times: user 6.31 s, sys: 1.16 s, total: 7.47 s
    Wall time: 3.54 s



The logistic regression models had middling accuracy compared to most of the other models.
The cross validation showed that overfitting was not greatly present with this model.
The model had similar rates of type 1 and type 2 error.
The model likely benefitted from being able to reduce the impact of parameters that were deemed to be excessively low.
This quality is especially useful when modeling text data because there's a large amount of words that wouldn't be used often enough to have a meaningful impact on the model, and words where there isn't a strong positive or negative connotation.


## Support Vector




```python
%%time

## train and fit model

svc = SVC().fit(x_train, y_train)

```

    CPU times: user 1h 13min 47s, sys: 10.4 s, total: 1h 13min 57s
    Wall time: 1h 14min 7s



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(svc.score(x_test, y_test))+'\n')

print("cross validation:\n" + str(cross_val_score(svc, x_test, y_test, cv=5))+'\n')

print("cross validation with AUC:\n" + str(cross_val_score(svc, x_test, y_test, cv=5, scoring='roc_auc'))+'\n')

print("confusion matrix:\n" + str(confusion_matrix(y_test, svc.predict(x_test)))+'\n')


```

    accuracy score:
    0.7168666666666667

    cross validation:
    [0.71941667 0.71308333 0.71408333 0.71175    0.71925   ]

    cross validation with AUC:
    [0.79124737 0.78816637 0.78936463 0.78624505 0.79545159]

    confusion matrix:
    [[21957  7848]
     [ 9140 21055]]

    CPU times: user 47min 50s, sys: 8.24 s, total: 47min 59s
    Wall time: 48min 5s



```python
%%time

## Classification Report

print(classification_report(y_test, svc.predict(x_test)))


```

                  precision    recall  f1-score   support

               0       0.71      0.74      0.72     29805
               1       0.73      0.70      0.71     30195

       micro avg       0.72      0.72      0.72     60000
       macro avg       0.72      0.72      0.72     60000
    weighted avg       0.72      0.72      0.72     60000

    CPU times: user 9min 16s, sys: 1.59 s, total: 9min 18s
    Wall time: 9min 20s


The support vector classifier had middling accuracy compared to most of the other models.
The cross validation showed that overfitting had very little prescence in in this model.
The model was marginally better at predicting the negative class.
The model relies on creating boundaries between datapoints that reflect different classes.
With text data, those boundaries are be harder to form due to the existence of words that show up a lot in poth positive and negative reviews.
As a result, the model relies more on its cost function to reduce error.


### Gradient Boost



```python
%%time

## train and fit model

cl = ensemble.GradientBoostingClassifier()

parameters = {
              'n_estimators': list(np.arange(200, 301, 50)),
              'max_depth': list(range(1,3)),
              'loss': ['deviance', 'exponential']
             }

acc_scorer = make_scorer(accuracy_score)

clf = GridSearchCV(cl, parameters, scoring=acc_scorer).fit(x_train,  y_train)

## Show Best Parameters
print(clf.best_params_)

```

    {'loss': 'deviance', 'max_depth': 2, 'n_estimators': 300}
    CPU times: user 1h 29min 25s, sys: 32.6 s, total: 1h 29min 58s
    Wall time: 1h 30min 6s



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(clf.score(x_test, y_test))+'\n')

print("cross validation:\n" + str(cross_val_score(clf, x_test, y_test, cv=5))+'\n')

print("cross validation with AUC:\n" + str(cross_val_score(clf, x_test, y_test, cv=5, scoring='roc_auc'))+'\n')

print("confusion matrix:\n" + str(confusion_matrix(y_test, clf.predict(x_test)))+'\n')

```

    accuracy score:
    0.71485

    cross validation:
    [0.71125    0.70925    0.71441667 0.706      0.7135    ]

    cross validation with AUC:
    [0.78811327 0.78279881 0.78677194 0.78261945 0.79045135]

    confusion matrix:
    [[21886  7919]
     [ 9190 21005]]

    CPU times: user 2h 29min 35s, sys: 58.2 s, total: 2h 30min 33s
    Wall time: 2h 30min 57s



```python
%%time

## Classification Report

print(classification_report(y_test, clf.predict(x_test)))

```

                  precision    recall  f1-score   support

               0       0.70      0.73      0.72     29805
               1       0.73      0.70      0.71     30195

       micro avg       0.71      0.71      0.71     60000
       macro avg       0.72      0.71      0.71     60000
    weighted avg       0.72      0.71      0.71     60000

    CPU times: user 302 ms, sys: 5.64 ms, total: 307 ms
    Wall time: 308 ms


The gradient boost model had middling accuracy compared to the other models.
The cross validation showed few signs of overfitting with this model.
The model was equally good at predicting both classes.
The strength of this model when it comes to making predictions using this data comes from its ability to reduce error over multiple iterations, while building on the strengths of the decision tree.

### Neural Network


```python
%%time

## train and fit model

mlp = MLPClassifier(hidden_layer_sizes=(100,)).fit(x_train, y_train)

```

    CPU times: user 19min 50s, sys: 2min 4s, total: 21min 54s
    Wall time: 5min 44s



```python
%%time

## Model Evaluation

print("accuracy score:\n" + str(mlp.score(x_test, y_test))+'\n')

print("cross validation:\n" + str(cross_val_score(mlp, x_test, y_test, cv=5))+'\n')

print("cross validation with AUC:\n" + str(cross_val_score(mlp, x_test, y_test, cv=5, scoring='roc_auc'))+'\n')

print("confusion matrix:\n" + str(confusion_matrix(y_test, mlp.predict(x_test)))+'\n')

print(classification_report(y_test, mlp.predict(x_test)))


```

    accuracy score:
    0.7371

    cross validation:
    [0.72091667 0.71683333 0.717      0.71675    0.715     ]

    cross validation with AUC:
    [0.79784296 0.79327906 0.79480397 0.79082147 0.79362804]

    confusion matrix:
    [[21792  8013]
     [ 7761 22434]]

                  precision    recall  f1-score   support

               0       0.74      0.73      0.73     29805
               1       0.74      0.74      0.74     30195

       micro avg       0.74      0.74      0.74     60000
       macro avg       0.74      0.74      0.74     60000
    weighted avg       0.74      0.74      0.74     60000

    CPU times: user 36min 33s, sys: 4min 23s, total: 40min 57s
    Wall time: 10min 34s


The neural network had high accuracy compared to the other models.
The cross validation showed few signs of overfitting with this model.
The model was equally good at predicting both classes.
This model outperformed the gradient boosting classifier and had significantly lower runtimes.

## Analysis and Conclusion

The random forest was by far the best model when it came to performing sentiment analysis on the customer reviews.
It is also important to note that the decision tree was the next best performing model, implying that the text data benefits greatly from the binary splitting processes that the decision tree undergoes.

Understanding how to better utilize supervised modeling techniques to perform customer sentiment analysis will give insight how to understand feedback on products.
Being able to act on these efficiently gathered insights, could result in strategic decisionmaking that can increase product quality.

This study established the best suprvised modeling technique for determining the sentiment of Amazon Reviews. The next step in using this data to gather insights from reviews would be to collect review data from different sources and use them to test the model. This would give insight as to how different types of text are percieved by the model. By being able to understand how supervised learning models can be affected by text from different contexts, the increased efficiency of sentiment analysis could result in more nuanced insights.
