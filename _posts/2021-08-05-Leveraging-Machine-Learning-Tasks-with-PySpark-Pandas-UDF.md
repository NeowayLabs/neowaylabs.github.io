---
title: "Leveraging Machine Learning Tasks with PySpark Pandas UDF"
categories:
  - data-science
tags:
  - pandas-udf
  - pyspark
excerpt: "Distributing Machine Learning Tasks using PySpark Pandas UDF"
---


Experimenting is the word that best defines the daily life of a Data Scientist. To build a decent machine learning model for a given problem, a Data Scientist needs to train several models. This process includes tasks such as finding optimal hyperparameters to the model, cross-validate models using K-fold, and sometimes even train a model that has several outputs. All of those tasks mentioned before are time-consuming and nonetheless extremely important for the success of the model development. In this blog post, we're going to show how PySpark Pandas UDF, a framework used to distribute python functions on Spark clusters, can be applied to enhance the Data Scientist's daily productivity.

# How does PySpark implement Pandas UDF (User Defined Function)?

As the name suggests, PySpark Pandas UDF is a way to implement User-Defined Functions (UDFs) in PySpark using Pandas DataFrame. The definition given by the PySpark API documentation is the following:

 > "Pandas UDFs are user-defined functions that are executed by Spark using Arrow to transfer data and Pandas to work with the data, which allows vectorized operations. A Pandas UDF is defined using the `pandas_udf` as a decorator or to wrap the function, and no additional configuration is required. A Pandas UDF behaves as a regular PySpark function API in general."

In this post, we are going to explore `PandasUDFType.GROUPED_MAP`, or in the latest versions of PySpark also known as `pyspark.sql.GroupedData.applyInPandas`. The main idea is straightforward,  Pandas UDF grouped data allow operations in each group of the dataset. Since grouped operations in spark are computed across the nodes of the cluster, we can manipulate our dataset in a way that allows different models to be computed in different nodes. Yes, my dudes... never underestimate the power of a `groupBy`. 

### Setting Up

Before getting into the specifics of applying Pandas UDF, let's set up the environment with some modules, global variables, and commonly used functions.

The first step is to import all the modules that are going to be used throughout this little experiment.


```python
import pandas as pd

from catboost import CatBoostClassifier

from itertools import product

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import (
    DoubleType, FloatType, IntegerType, StringType, StructField, StructType
)

from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
```

And set some global variables that are going to be used multiple times.

```python
N_FEATURES = 20
N_CLASSES = 10
```

A common step in every task explored in this notebook is the training and evaluation of a machine learning model. This step is encapsulated in the following function, which trains and evaluates a CatBoost model  based on its accuracy score.


```python
def train_and_evaluate_model(X_train, y_train, X_test, y_test, kwargs={}):

    # split data
    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # create model
    model = CatBoostClassifier(
        nan_mode='Min',
        random_seed=42,
        boosting_type='Plain',
        bootstrap_type='Bernoulli',
        rsm=0.1,
        loss_function='Logloss',
        use_best_model=True,
        early_stopping_rounds=100,
        **kwargs
    )

    # fit model
    model.fit(X_train.values, y_train.values, eval_set=(X_eval, y_eval))
    
    # evaluate model
    accuracy = accuracy_score(model.predict(X_test), y_test)

    return accuracy
```

To train and test our CatBoost model, we will also need some data. So let's create our dataset using scikit-learn's `make_multilabel_classification` function and build our PySpark DataFrame from it.


```python
X, y = make_multilabel_classification(
    n_samples=10000,
    n_features=N_FEATURES,
    n_classes=N_CLASSES,
    random_state=42
)
```


```python
pdf = pd.DataFrame(X)
for i in range(N_CLASSES):
    pdf[f'y_{i}'] = y[:, i]
df = spark.createDataFrame(pdf)
```


```python
print(f'number of rows in the dataset: {df.count()}')
```

    number of rows in the dataset: 10000



```python
df.limit(5).toPandas()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>y_0</th>
      <th>y_1</th>
      <th>y_2</th>
      <th>y_3</th>
      <th>y_4</th>
      <th>y_5</th>
      <th>y_6</th>
      <th>y_7</th>
      <th>y_8</th>
      <th>y_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>



Finally, for a more efficient Spark computation, we're going to enable arrow-based columnar data transfer.


```python
spark.conf.set('spark.sql.execution.arrow.enabled', 'true')
```

## Distributed Grid Search

In machine learning, hyperparameters are parameters whose values are used to control the model's architecture and its learning process. Oftentimes when training a model you need to optimize these hyperparameters but, despite the ability of ML to find optimal internal parameters and thresholds for decisions, hyperparameters are set manually. 

If the search space contains too many possibilities, you'll need to spend a good amount of time testing to find the best combination of hyperparameters. A way to accelerate this task is to distribute the search process on the nodes of a Spark cluster.

One question that arises with this approach is: "Ok, but I'm using an algorithm that hasn't been implemented on Spark yet, how can I distribute this process with these limitations?" Don't worry! That's a question we are here to answer!

First, we have to define the hyperparameter search space. For that, we are going to create an auxiliary PySpark DataFrame where each row is a unique set of hyperparameters.


```python
values_range = list(
    product(
        [200, 210, 220, 230, 240, 250, 260, 270, 280, 290],
        [3, 4, 5, 6, 7],
        [0.02, 0.07, 0.1, 0.15, 0.2],
        ['MinEntropy', 'Uniform', 'UniformAndQuantiles', 'GreedyLogSum'],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [0.5, 0.6, 0.7, 0.8],
    )
)

schema = StructType(
    [
        StructField('iterations', IntegerType(), True),
        StructField('depth', IntegerType(), True),
        StructField('learning_rate', DoubleType(), True),
        StructField('feature_border_type', StringType(), True),
        StructField('l2_leaf_reg', FloatType(), True),
        StructField('subsample', FloatType(), True)
    ]
)

df_grid = spark.createDataFrame(data=values_range, schema=schema)
df_grid = df_grid.withColumn('replication_id', sf.monotonically_increasing_id())
```

```python
df_grid.limit(5).toPandas()
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
      <th>iterations</th>
      <th>depth</th>
      <th>learning_rate</th>
      <th>feature_border_type</th>
      <th>l2_leaf_reg</th>
      <th>subsample</th>
      <th>replication_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>200</td>
      <td>4</td>
      <td>0.1</td>
      <td>Uniform</td>
      <td>2.0</td>
      <td>0.5</td>
      <td>171798691840</td>
    </tr>
    <tr>
      <th>1</th>
      <td>200</td>
      <td>4</td>
      <td>0.1</td>
      <td>Uniform</td>
      <td>2.0</td>
      <td>0.6</td>
      <td>171798691841</td>
    </tr>
    <tr>
      <th>2</th>
      <td>200</td>
      <td>4</td>
      <td>0.1</td>
      <td>Uniform</td>
      <td>2.0</td>
      <td>0.7</td>
      <td>171798691842</td>
    </tr>
    <tr>
      <th>3</th>
      <td>200</td>
      <td>4</td>
      <td>0.1</td>
      <td>Uniform</td>
      <td>2.0</td>
      <td>0.8</td>
      <td>171798691843</td>
    </tr>
    <tr>
      <th>4</th>
      <td>200</td>
      <td>4</td>
      <td>0.1</td>
      <td>Uniform</td>
      <td>3.0</td>
      <td>0.5</td>
      <td>171798691844</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f'number of different hyperparameter combinations: {df_grid.count()}')
```

    number of different hyperparameter combinations: 24000


For each hyperparameter row, we want to replicate our data so we can later process every hyperparameter set individually.


```python
df_replicated = df.crossJoin(df_grid)
```


```python
print(f'number of rows in the replicated dataset: {df_replicated.count()}')
```

    number of rows in the replicated dataset: 240000000


The last step is to specify how each Spark node will handle the data. To do that, we define the `run_model` function. It basically extracts the hyperparameters and the data from the input Spark DataFrame, then trains and evaluates the model, returning its results.


```python
# declare the schema for the output of our function
schema = StructType(
    [
        StructField('replication_id', IntegerType(),True),
        StructField('accuracy', FloatType(),True),
        StructField("iterations", IntegerType(), True),
        StructField("depth", IntegerType(), True),
        StructField("learning_rate", DoubleType(), True),
        StructField("feature_border_type", StringType(), True),
        StructField("l2_leaf_reg", FloatType(), True),
        StructField("subsample", FloatType(), True)
     ]
)

# decorate our function with pandas_udf decorator
@pandas_udf(schema, sf.PandasUDFType.GROUPED_MAP)
def hyperparameter_search(pdf):

    # get hyperparameter values
    kwargs = {
        'iterations': pdf.iterations.values[0],
        'depth': pdf.depth.values[0],
        'learning_rate': pdf.learning_rate.values[0],
        'feature_border_type': pdf.feature_border_type.values[0],
        'l2_leaf_reg': pdf.l2_leaf_reg.values[0],
        'subsample': pdf.subsample.values[0]
    }
    
    # get data and label
    X = pdf[[str(i) for i in range(N_FEATURES)]]
    y = pdf['y_0']

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # get accuracy
    accuracy = train_and_evaluate_model(X_train, y_train, X_test, y_test, kwargs)

    # return results as pandas DF
    kwargs.update({
        'replication_id': pdf.replication_id.values[0],
        'accuracy': accuracy
    })
    results = pd.DataFrame([kwargs])

    return results
```

We can now group the Spark Dataframe by the `replication_id` and apply the `run_model` function. This way, every hyperparameter combination will be used to train a different model in a distributed system.


```python
results = df_replicated.groupby('replication_id').apply(hyperparameter_search)
```


```python
%%time

results.sort('accuracy', ascending=False).limit(5).toPandas()
```

    CPU times: user 11.6 s, sys: 13.5 s, total: 25.1 s
    Wall time: 29min 10s





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
      <th>replication_id</th>
      <th>accuracy</th>
      <th>iterations</th>
      <th>depth</th>
      <th>learning_rate</th>
      <th>feature_border_type</th>
      <th>l2_leaf_reg</th>
      <th>subsample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24</td>
      <td>0.9145</td>
      <td>210</td>
      <td>7</td>
      <td>0.20</td>
      <td>Uniform</td>
      <td>6.0</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>0.9125</td>
      <td>250</td>
      <td>3</td>
      <td>0.20</td>
      <td>Uniform</td>
      <td>2.0</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13</td>
      <td>0.9125</td>
      <td>230</td>
      <td>6</td>
      <td>0.15</td>
      <td>MinEntropy</td>
      <td>3.0</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11</td>
      <td>0.9125</td>
      <td>290</td>
      <td>3</td>
      <td>0.20</td>
      <td>Uniform</td>
      <td>5.0</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>0.9125</td>
      <td>220</td>
      <td>3</td>
      <td>0.10</td>
      <td>MinEntropy</td>
      <td>6.0</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>



With this distributed approach, we were able to run 24000 combinations of hyperparameters in only 29 minutes.

## Distributed K-Fold Cross-Validation

Having an optimal set of hyperparameters, another important task is to perform a K-Fold Cross-Validation of your model to prevent (or minimize) the undesired effects of overfitting. The more folds you add to this experiment the more robust your model will be. However, you'll have to spend more time  training models for each fold. Once again, a way to avoid the time trap is to use Spark and compute each fold in an individual node of your Spark cluster. 

We perform this in a very similar manner to how we distribute the grid-search, the difference being that we replicate our dataset according to the number of folds. So if our cross-validation uses 8 folds, our dataset will be replicated 8 times.

Here, our first step is to define the number of folds we want to cross-validate our model.


```python
N_FOLDS = 8
```

Following this, we define some code to randomly split our dataset according to the number of folds defined above.


```python
proportion = 1 / N_FOLDS
splits = df.randomSplit([proportion] * N_FOLDS, 42)
df_folds = splits[0].withColumn('fold', sf.lit(0))
for i in range(1, N_FOLDS):
    df_folds = df_folds.union(
        splits[i].withColumn('fold', sf.lit(i))
    )
```

After the split, we replicate the dataset K times. 


```python
df_numbers = spark.createDataFrame(
    pd.DataFrame(list(range(N_FOLDS)),columns=['replication_id'])
)
```


```python
df_numbers.toPandas()
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
<table border="1" class="dataframe" style="width:150px">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>replication_id</th>
    </tr>
  </thead>
  <tbody style="text-align: right;">
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_replicated = df_folds.crossJoin(df_numbers)
```


```python
print(f'number of rows in the replicated dataset: {df_replicated.count()}')
```

    number of rows in the replicated dataset: 80000


Here we have another difference compared to the grid search approach. In the function below, we define the train and test datasets according to a `replication_id` and a `fold_id`. If the `replication_id` is equal to the `fold_id`, we set that fold as the test fold while the rest of the folds are used as the training set.


```python
# declare the schema for the output of our function
schema = StructType(
    [
        StructField('replication_id', IntegerType(), True),
        StructField('accuracy', FloatType(), True)
    ]
)

# decorate our function with pandas_udf decorator
@pandas_udf(schema, sf.PandasUDFType.GROUPED_MAP)
def cross_validation(pdf):
    
    # get repliaction id
    replication_id = pdf.replication_id.values[0]
    
    # get data and label
    columns = [str(i) for i in range(N_FEATURES)]
    X_train = pdf[pdf.fold != replication_id][columns]
    X_test = pdf[pdf.fold == replication_id][columns]
    y_train = pdf[pdf.fold != replication_id]['y_0']
    y_test = pdf[pdf.fold == replication_id]['y_0']

    # get accuracy
    accuracy = train_and_evaluate_model(X_train, y_train, X_test, y_test)

    # return results as pandas DF
    results = pd.DataFrame([{
        'replication_id': replication_id,
        'accuracy': accuracy
    }])

    # save the model (if you want to retrieve it later)

    return results
```

One thing that you might have to take into account with this approach is how to save each trained model since each model is trained in a different node. To do this, depending on your cloud provider, you can use some python library developed to transfer files from the cluster nodes directly to a cloud bucket (like Google Cloud Storage or Amazon S3). However, if you're only interested in the performance of the cross-validated model, the function above is enough.


```python
results = df_replicated.groupby('replication_id').apply(cross_validation)
```


```python
%%time

results.sort('accuracy', ascending=False).toPandas()
```

    CPU times: user 1.03 s, sys: 1.24 s, total: 2.27 s
    Wall time: 35.9 s





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
<table border="1" class="dataframe" style="width:200px">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>replication_id</th>
      <th>accuracy</th>
    </tr>
  </thead>
  <tbody style="text-align: right;">
    <tr>
      <th>0</th>
      <td>4</td>
      <td>0.900715</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>0.895292</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.893720</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0.893601</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.891801</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>0.890048</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0.883293</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6</td>
      <td>0.882946</td>
    </tr>
  </tbody>
</table>
</div>



In this experiment, we evaluated 8 folds (one in each node of the cluster) in only 35 seconds. And the best fold (number 4) reached an accuracy score of 0.900.

## Distributed Multiple Output Model

Following the same philosophy, we can take advantage of PySpark Pandas UDF to distribute the training of multi-output models. For this task, we have a set of features and a set of labels, where we must train a model for each label with the same training data.

Some packages like `scikit-learn` have already implemented this approach to Random Forest algorithms. `CatBoost` also has the option of training with multi-outputs. However, these implementations have limited hyperparameter and loss-function options compared to the single output API.  Considering this, Pandas UDF is an alternative to automate the training of multiple models at once, using all the options that any other machine learning library usually offers to single output model training.

Since our dataset has multiple label columns, the approach this time is to pivot our data in a way we can replicate the data for each specific label. So we create a column to map each label and append all the labels in one single label column as shown below:


```python
features = [f'{i}' for i in range(N_FEATURES)]
targets = [f'y_{i}' for i in range(N_CLASSES)]

df_multipe_output = df.select(
    *features,
     sf.lit(targets[0]).alias('y_group'),
     sf.col(targets[0]).alias('Y')
)
for target in targets[1:]:
    df_multipe_output = df_multipe_output.union(
        df.select(
            *features,
            sf.lit(target).alias('y_group'),
            sf.col(target).alias('Y')
        )
    )
```


```python
print(f'number of rows in the dataset: {df_multipe_output.count()}')
```

    number of rows in the dataset: 100000



```python
df_multipe_output.limit(5).toPandas()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>y_group</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>y_0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>y_0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>y_0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>y_0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>y_0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



Having defined our spark multi-output dataset we are ready to define the function to perform the model training. 


```python
# declare the schema for the output of our function
schema = StructType(
    [
        StructField('y_group', StringType(), True),
        StructField('accuracy', FloatType(), True)
    ]
)

# decorate our function with pandas_udf decorator
@pandas_udf(schema, sf.PandasUDFType.GROUPED_MAP)
def multi_models(pdf):
    
    # get group
    y_group = pdf.y_group.values[0]
    
    # get data and label
    X = pdf.drop(['Y', 'y_group'], axis=1)
    y = pdf['Y']

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # get accuracy
    accuracy = train_and_evaluate_model(X_train, y_train, X_test, y_test)

    # return results as pandas DF
    results = pd.DataFrame([{
        'y_group': y_group,
        'accuracy': accuracy
    }])

    return results
```

Once everything is set, you can call the groupBy method on the `y_group` column to distribute the training of each model.


```python
results = df_multipe_output.groupby('y_group').apply(multi_models).orderBy('accuracy')
```


```python
%%time

results.sort('accuracy', ascending=False).limit(5).toPandas()
```

    CPU times: user 193 ms, sys: 195 ms, total: 388 ms
    Wall time: 9.24 s





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
<table border="1" class="dataframe" style="width:200px">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y_group</th>
      <th>accuracy</th>
    </tr>
  </thead>
  <tbody style="text-align: right;">
    <tr>
      <th>0</th>
      <td>y_6</td>
      <td>0.9740</td>
    </tr>
    <tr>
      <th>1</th>
      <td>y_4</td>
      <td>0.9330</td>
    </tr>
    <tr>
      <th>2</th>
      <td>y_5</td>
      <td>0.9325</td>
    </tr>
    <tr>
      <th>3</th>
      <td>y_8</td>
      <td>0.8990</td>
    </tr>
    <tr>
      <th>4</th>
      <td>y_0</td>
      <td>0.8910</td>
    </tr>
  </tbody>
</table>
</div>



# Conclusion

In this post, we showed some examples of how PySpark Pandas UDF can be used to distribute processes involving the training of machine learning models. Some of the approaches showed can be used to save time or to run experiments on a larger scale that would otherwise be too memory-intensive or prohibitively expensive.

We hope this can be useful and that you guys enjoyed the content. Please leave your comments and questions below.

See you in the next post. That's all, folks.


*Authors: [Igor Siqueira Cortez](https://www.linkedin.com/in/igor-cortez-56793825/), [Vitor Hugo Medeiros De Luca](https://www.linkedin.com/in/vitordeluca/), [Luiz Felipe Manke](https://br.linkedin.com/in/luizmanke/)*
{: .notice}