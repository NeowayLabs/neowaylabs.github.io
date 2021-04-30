---
title: "Shapley Values at Scale"
categories:
  - data-science
tags:
  - shapley-values
  - pyspark
excerpt: "Scalling Shapley values calculation"
---

The Shapley values is a solution concept from cooperative game theory introduced by Lloyd Shapley in 1951, who later was awarded with the Nobel Prize in Economics in 2012. His theory was developed to understand how the surplus from a coalition (eg. a set of business man who decide to run a business) could be optimally distributed given that some members contribute more or less than the others. In summary, the theory states that each player in a coalition is worth it's average marginal contribution under all possible coalitions that this player might participate.

With the rise of concerns about privacy and fairness about decisions taken using algorithms, the issue of interpretability became more popular. What economists couldn't imagine is that this concept would revolutionize the way of how data scientists interpret machine learning models today. Machine learning algorithms are functions dedicated to prediction tasks and the more precise it gets the more complex and less interpretable becomes. Think of regression models and decision trees as the simplest models and ensemble and deep neural networks as the most complex.

In the context of Shapley values, algorithms are the coalitions and the features/variables of the model are the members/players of this cooperative game. Compared to the traditional feature importance methods such as Information Gain and Gini Index, that offer insights about the relevance of a feature, the Shapley values methodology goes further and adds insights on feature relevance, how each feature impacts the prediction of an individual data point and how the feature on average impacts the outcome of the model. Nonetheless, given the concept involves complex computation, the calculations on datasets with high volume of data can become tedious and sometimes simply not feasible.

The focus of this blog post is not to dig deep in these concepts, but to show how it's possible to scale the interpretability of a black box model, specifically using CatBoost, PySpark and Pandas UDF. In the end, we will have answered the following questions:

* Is it possible to scale the Shapley values for each point in a large dataset so that we can interpret each prediction individually?
* Is there a gain in calculation time if we scale/distribute data with Spark?
* Is there any difference between the Shapley values obtained from distributed vs centralized data?

So, let's get started!

# Defining the dataset

Let's start by creating a synthetic dataset. Scikit-learn provides a very good API for creating a dataset to be used in a classification problem. This API has many options (we encourage you to check these options) and, among them, we chose the following ones to generate a dataset with a milion rows, ten features (seven of which are informative) and 2-class target variable.

```python
X, y = datasets.make_classification(n_samples=1000000,
                                    n_features=10,
                                    n_informative=7,
                                    n_classes=2,
                                    random_state=123)
```

This dataset will be divided into training (80%) and testing (20%).

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
```

# Training the model

Now, the data will be used to create the Pool object needed to train the CatBoost Classifier model.

```python
train_c = Pool(data=X_train, label=y_train)
test_c = Pool(data=X_test, label=y_test)
```

Here, the CatBoost Classifier is trained using hyperparameters that speed up the execution of the training. In this article, we will not focus on the tuning of these parameters, nor on the performance of the model.

```python
model_c = CatBoostClassifier(iterations=1000,
                             random_seed=123,
                             boosting_type="Plain",
                             bootstrap_type="Bernoulli",
                             rsm=0.1,
                             loss_function='Logloss',
                             use_best_model=True,
                             early_stopping_rounds=50)

model_c.fit(train_c,
            eval_set=test_c,
            plot=True,
            verbose=False)
```

<figure class="align-center">
  <img src="/images/posts/shapley-values-at-scale/catboost-training.png" alt="">
  <figcaption>CatBoost Training.</figcaption>
</figure>

Just as a check, let's see the final accuracy of the model.


```python
accuracy = accuracy_score(model_c.predict(X_test), y_test)
print(accuracy)
```

    0.934085


# Calculating the Feature Importance

Feature importance is one of the most common and simple methods of model interpretability. In order to have a fair comparison between the methods here presented, the dataset used for the calculations will be the complete data set. This option was chosen so that the calculation times were longer and less subject to inaccuracy of the measurements.


```python
%%time
ft_importance = model_c.get_feature_importance(Pool(X, y), prettified=True)
```

    CPU times: user 6.18 s, sys: 65.5 ms, total: 6.24 s
    Wall time: 6.17 s


```python
sns.barplot(x=ft_importance['Feature Id'],
            y=ft_importance['Importances'])
```
    
<figure class="align-center">
  <img src="/images/posts/shapley-values-at-scale/feature_importance_by_order.png" alt="">
  <figcaption>Feature Importances.</figcaption>
</figure>

By getting the default type of feature importance from a Catboost model, it is possible to observe that the calculation time was relatively short, as expected due to the simplicity of the method. The three most important features are 7, 4 and 3 respectively.

# Calculating the Shapley Values

Finally we come to the calculation of the shapley values. The API used is the same for the Feature Importance (**get_feature_importance**), just adding the parameter **type="ShapValues"**. As a first observation, the time to calculate is almost twice as long. This is expected given the complexity of the calculation.

```python
%%time
shap_values = model_c.get_feature_importance(Pool(X, y), type="ShapValues")
```

    CPU times: user 2min 33s, sys: 2.76 s, total: 2min 36s
    Wall time: 14 s


The variable **shap_values** is a numpy matrix where the last column is composed by equal elements that represent the expected value. If you need further information about this, check out this link [here](https://catboost.ai/docs/concepts/shap-values.html). In other words, the Shapley Values will be all elements of this numpy matrix but the last column.


```python
expected_value = shap_values[0,-1]
shap_values = shap_values[:,:-1]
```

The summary plot of the shapley values is shown in the following figure. It is possible to see that there is a correlation between the feature importance and the shapley values, but they are not the same and this is already expected as they are different approaches. For example, the three main features that have impact on model output are respectively 4, 7 and 3 for the shapley values and 7, 4 and 3 for the features importance.


```python
shap.summary_plot(shap_values, X)
```
   
<figure class="align-center">
  <img src="/images/posts/shapley-values-at-scale/shap_centralized_regular.png" alt="">
  <figcaption>Shapley Values using calculation type regular (default).</figcaption>
</figure>

Another cool property of shapley values is that the sum of feature contributions are equal to the value prediction. Let's check this out. Here is the sum of feature contributions for the first object in our dataset

```python
sum(shap_values[0])
```
    2.366528338073737

And here is the raw value of prediction for the first object of our dataset

```python
model_c.predict(X, prediction_type = 'RawFormulaVal')[0] 
```
    2.366528338073735

As we can see, they are not perfectly equal, but the difference between the measures are almost near zero.


# Calculating the Approximate Shapley Values

Another interesting approach when calculating shapley values is to use the option **shap_calc_type = "Approximate"**. This makes the calculation of shapley values faster and can be useful in cases where the amount of features and data is very large. For this example, as the amount of data is relatively small, a slight difference is noticed in the calculation time.

```python
%%time
shap_aprox = model_c.get_feature_importance(Pool(X, y), type="ShapValues", shap_calc_type="Approximate")
```
    CPU times: user 2min 36s, sys: 2.51 s, total: 2min 38s
    Wall time: 13.7 s

```python
expected_value_aprox = shap_aprox[0,-1]
shap_aprox = shap_aprox[:,:-1]
```

The summary plot of the approximate shapley values is shown in the following figure. As a consequence of the faster calculation, there are minor distortions in the calculated shapley values. A close look at this figure shows small differences from the previous summary plot where the option **shap_calc_type = "Approximate"** was not used. For example, features 0, 2, 6 and 8 are not in the same position.


```python
shap.summary_plot(shap_aprox, X)
```
    
<figure class="align-center">
  <img src="/images/posts/shapley-values-at-scale/shap_centralized_approximate.png" alt="">
  <figcaption>Shapley Values using calculation type approximate.</figcaption>
</figure>


# Shapley Values At Scale

So let's see the magic! The first step is to create a Spark dataframe using the X values.


```python
spark_df = spark.createDataFrame(pd.DataFrame(X))
```

The following Pandas UDF is one of the main tips in this article. The API **get_feature_importance** is "embedded" in the function **shap_calc** that will be used by the PySpark function **withColumn()** to create a new column in the Spark dataframe containing the calculated shapley values in a distributed way.


```python
@pandas_udf(returnType=ArrayType(DoubleType()))
def shap_calc(*cols):
    X = pd.concat(cols, axis=1).values
    shap = model_c.get_feature_importance(
        data=Pool(X),
        fstr_type="ShapValues"
    )
    return pd.Series(shap.tolist())
```

As mentioned, in this step the column **shap_array** will be created containing the calculated shapley values.


```python
spark_df = spark_df.withColumn('shap_array', shap_calc(*model_c.feature_names_))
```

Since the function **withColumn** is lazy, let's perform a simple non-lazy operation to get an idea of the necessary time to calculate the shapley values in this case and compare it with the previous numbers. As expected, the time required to perform the calculation is shorter than the centralized calculations performed previously. In addition, this difference could be even greater if the number of workers used was greater. In this example, eight workers are being used.


```python
%%time
spark_df.cache().count()
```

    CPU times: user 7.64 ms, sys: 4.06 ms, total: 11.7 ms
    Wall time: 8.22 s

    1000000

Likewise, the option **shap_calc_type = "Approximate"** can also be used here. Another Pandas UDF will be created by adding this parameter and will be called **shap_calc_approx**.

```python
@pandas_udf(returnType=ArrayType(DoubleType()))
def shap_calc_approx(*cols):
    X = pd.concat(cols, axis=1).values
    shap_v = model_c.get_feature_importance(
        data=Pool(X),
        fstr_type="ShapValues",
        shap_calc_type="Approximate",
    )
    return pd.Series(shap_v.tolist())
```

In this step, another column **shap_array_approx** will be created in the Spark dataframe containing the calculated approximate shapley values.


```python
spark_df = spark_df.withColumn('shap_array_approx', shap_calc_approx(*model_c.feature_names_))
```

Let's look at the time for this calculation. As expected, this time is even shorter for the approximate mode.

```python
%%time
spark_df.cache().count()
```

    CPU times: user 3.85 ms, sys: 5.06 ms, total: 8.91 ms
    Wall time: 3.84 s

    1000000

For calculation purposes, the purpose of this article would be closed here. However, there remains a graphical visualization of how these calculated shapley values behave in relation to the centralized version previously shown. 

To do it, the first step is to "explode" the calculated shapley values into columns. The code below will create a Spark dataframe with 30 columns. The first 10 are the X columns. The next 10 columns are "exploded" from the original **shap_array** column. The last 10 columns are "exploded" from the original **shap_array_approx** column.


```python
feat_size = X.shape[1]
feat_index = range(feat_size)

df_with_shap_values = spark_df.select(
    *[sf.col(str(c)).alias(f'Feature {c}') for c in feat_index], # feature cols
    *[sf.col('shap_array').getItem(c).alias(f"SHAP {c}") for c in feat_index], # SHAP for each feature col
    *[sf.col('shap_array_approx').getItem(c).alias(f"SHAP APPROX {c}") for c in feat_index], # SHAP APPROX for each feature col
)
```

The next step is to centralize all of these values by transforming them to Numpy and then getting the respective columns.


```python
np_values = df_with_shap_values.toPandas().to_numpy()

feat_values = np_values[:, :10]
shap_values = np_values[:, 10:20]
shap_values_approx = np_values[:, 20:]
```

Finally, it is possible to visually observe that the summary plots are different from each other, just as they were in the centralized version, however they are the same when compared to their respective centralized versions.


```python
shap.summary_plot(shap_values, feat_values)
```
    
<figure class="align-center">
  <img src="/images/posts/shapley-values-at-scale/shap_distributed_regular.png" alt="">
  <figcaption>Shapley Values distributed using calculation type regular (default).</figcaption>
</figure>
    
```python
shap.summary_plot(shap_values_approx, feat_values)
```
    
<figure class="align-center">
  <img src="/images/posts/shapley-values-at-scale/shap_distributed_approximate.png" alt="">
  <figcaption>Shapley Values distributed using calculation type approximate.</figcaption>
</figure>


# Conclusions

This article started by making a brief comparison between the Feature Importance and the Shapley Values. They are similar and correlated but not equal. Following, this article showed four calculation approaches for Shapley Values:

* 1st - Centralized using calculation default (regular)
* 2nd - Centralized using calculation approximate
* 3rd - Distributed using calculation default (regular)
* 4th - Distributed using calculation approximate

It can be seen that the time to calculate the Shapley Values decreases after each approach (from the first to the fourth) with the fourth approach being the fastest. On the other hand, there is a cost to this speed gain: the accuracy of the calculated values drops slightly.

Finally, it is also possible to observe that similar calculation approaches (whether centralized or distributed) have the same calculated values. In this way, the first and third approaches produce the same values, as do the second and fourth approaches.



*Authors: [Fernando Felix](https://www.linkedin.com/in/fernandofnjr/), [Igor Siqueira Cortez](https://www.linkedin.com/in/igor-cortez-56793825/), [Vitor Hugo Medeiros De Luca](https://www.linkedin.com/in/vitordeluca/)*
{: .notice}
