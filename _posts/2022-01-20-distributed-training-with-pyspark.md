---
title: "Distributed Training with PySpark"
categories:
  - data-science
tags:
  - distributed-training
  - pyspark
  - ensemble
excerpt: ""
---

There are several definitions around the term "Big Data" all over the internet, books, and so on. In practical terms, a data scientist "feels" a Big Data scenario every time the amount of data does not fit in memory. From this point on, there are several tools available on the internet that try to solve this problem, but one of the most common ways would be to work with PySpark.

Once PySpark is in use and the data is distributed across a cluster, various functions can be used to transform the data. However, the strategy on how to train your AI model may not yet be defined. Several options are available. Let's try to list them:

1. Centralize all data to train the model;
2. Train in a fully distributed way;
3. Centralize a representative sample of the data;
4. Train on multiple samples and then ensemble them.

Option 1 will be discussed only for the sake of benchmarking, as it would not be possible by the definition of the problem presented by this post. Options 2, 3 and 4 will be shown with their pros and cons.

Before starting, we strongly recommend reading the post "[Leveraging Machine Learning Tasks with PySpark Pandas UDF](https://neowaylabs.github.io/data-science/Leveraging-Machine-Learning-Tasks-with-PySpark-Pandas-UDF/)", written by the same authors of this study.

## Let's start importing the libraries


```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from catboost_spark import CatBoostClassifier as CatBoostClassifierSpark
from catboost_spark import Pool as PoolSpark
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import *
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Seaborn config
sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(17,5)})
```

## Defining the synthetic dataset

For this experiment, we are going to use a synthetic dataset using sklearn `datasets.make_classification` to generate the data. The dataset will contain one million instances and 20 features. This number could be even higher for the distributed approach, but we need to compare it to the centralized approach and therefore we cannot increase it that much.

In order to make things a bit harder, we are going to configure only 10 features as informative, flip 10% of the target values and short the distance between classes. Why this? Because we choosed this way. You are welcome to reproduce this experiment with other combinations. Have fun!


```python
N_SAMPLES = 1_000_000
N_FEATURES = 20

X, y = datasets.make_classification(n_samples=N_SAMPLES,
                                    n_features=N_FEATURES,
                                    n_informative=10,
                                    n_redundant=5,
                                    n_repeated=5,
                                    n_classes=2,
                                    n_clusters_per_class=4,
                                    flip_y=0.1,
                                    class_sep=0.75,
                                    random_state=123,
                                    shuffle=True)
```

## Train-Test split

In this step, the data are divided into training (70%) and testing (30%). Then the training data is split into 5 random folds for cross-validation training. Cross-validation is implemented to ensure that results are not biased by a single random choice of training data.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

N_FOLDS = 5

skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=123)

folds_index = list(skf.split(X_train, y_train))
```

## First approach: centralized training

In the first approach, we are going to use a centralized CatBoost model with the default hyperparameters set. The metric we are going to use to compare the approachs is the accuracy since the target is balanced and we do not have any business considerations here.

One model is expected for each fold reaching a total of 5 models. For each model, 4 folds are used for training and 1 fold used for validation. This is shown below.


```python
ctb_models = [CatBoostClassifier() for i in range(N_FOLDS)]

for i, (train_index, valid_index) in tqdm(enumerate(folds_index)):

    X_fold_train, X_valid = X_train[train_index], X_train[valid_index]
    y_fold_train, y_valid = y_train[train_index], y_train[valid_index]

    ctb_models[i].fit(X_fold_train,
                      y_fold_train,
                      eval_set=[(X_valid, y_valid)],
                      verbose=0,
                      plot=False, # Turn True to see plots
                      use_best_model=True,
                      early_stopping_rounds=100)
```

    5it [03:51, 46.36s/it]


Since we have 5 trained models, each model makes a prediction on the test set and an average among all models shows the final prediction.


```python
centralized_preds = np.zeros_like(y_test)

for model in tqdm(ctb_models):
    preds = model.predict_proba(X_test)[:,1]
    centralized_preds = centralized_preds + preds/N_FOLDS
    
accuracy_score(y_test, np.rint(centralized_preds))
```

    0.9045



Accuracy reached 0.9045. This number alone doesn't say anything because we're interested in making comparisons with upcoming approaches. So it is a benchmark number. 

Before continuing reading, try to think about what the expected results would be and only after continuing reading.

## Second approach: distributed training

In the second approach, we are going to use a distributed CatBoost model with the default hyperparameters set. The same cross-validation is performed as done in the first approach, but the data always keeps distributed across the entire cluster.

As the synthetic dataset created is initially centralized, we need the following steps so the models can be trained:

- Convert training and validation data to Pandas dataframe;
- Convert Pandas dataframe to Spark dataframe (data is distributed to the cluster here);
- Convert Spark dataframe to MLlib format
- Convert MLlib format to CatBoost Pool format

Multiple conversions, right? Right, but realize that in a real scenario the data would already be in a Spark format.


```python
from pyspark.ml.feature import VectorAssembler

spark_models = [CatBoostClassifierSpark() for i in range(N_FOLDS)]
columns = [f'c{i}' for i in range(0, N_FEATURES)]
classifier_fitted = []

assembler = VectorAssembler(inputCols=columns,
                            outputCol="features")

for i, (train_index, valid_index) in tqdm(enumerate(folds_index)):

    # Get datasets according to folds index
    X_fold_train, X_valid = X_train[train_index], X_train[valid_index]
    y_fold_train, y_valid = y_train[train_index], y_train[valid_index]
    
    # Convert to Pandas DF
    df_train = pd.DataFrame(X_fold_train, columns=columns)
    df_train['label'] = y_fold_train

    df_valid = pd.DataFrame(X_valid, columns=columns)
    df_valid['label'] = y_valid
    
    # Convert to Spark DF
    dfs_train = spark.createDataFrame(df_train)
    dfs_valid = spark.createDataFrame(df_valid)

    # Convert to MLlib format
    dfs_train_fit = assembler.transform(dfs_train).select("features", "label")
    dfs_valid_fit = assembler.transform(dfs_valid).select("features", "label")
    
    # Convert to Pool format
    trainPool = PoolSpark(dfs_train_fit)
    evalPool = PoolSpark(dfs_valid_fit)
    
    classifier_fitted.append(spark_models[i].fit(trainPool, [evalPool]))
```

    5it [21:21, 256.21s/it] 

As with the first centralized approach, we have 5 trained models and need to perform the predictions on the test set. An average among all models shows the final prediction, but in this case we need to perform all the conversions, just like we did in the training step.


```python
distributed_preds = np.zeros_like(y_test)

# Convert to Pandas DF
df_test = pd.DataFrame(X_test, columns=columns)
df_test['label'] = y_test

# Convert to Spark DF
dfs_test = spark.createDataFrame(df_test)

# Convert to MLlib format
dfs_test_fit = assembler.transform(dfs_test).select("features", "label")

# Convert to Pool format
testPool = PoolSpark(dfs_test_fit)

for model in tqdm(classifier_fitted):
    preds = model.transform(testPool.data).select("probability").toPandas()["probability"].str[1]
    distributed_preds = distributed_preds + preds/N_FOLDS
    
accuracy_score(y_test, np.rint(distributed_preds))
```

    0.9043466666666666

Accuracy reached 0.9043. Note that this accuracy is very close to the accuracy obtained with the first centralized approach (0.9045). The difference is in the execution time: distributed training takes longer (21:21 minutes), approximately 5.5 times longer for this case shown. 

Also, we did the following experiment: we ran this experiment for 29 different dataset sizes (e.g. 500k rows, 100k rows, etc) and the RMSE between centralized and distributed approaches was 0.00234! This is a very small number compared to the accuracies obtained. 

## Third approach: centralized training on a sample of the dataset

Another possible solution deals with samples. Each of the 5 folds will be considered as a sample. Then these samples will be sent to the nodes via a Pandas UDF. In the code below, the Pandas Dataframe receives a new column called "fold" and the number in this column represents which fold this row belongs to.


```python
# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# Convert to Pandas DF
df_train = pd.DataFrame(X_train, columns=columns)
df_train['label'] = y_train

# Insert 'fold' column
for i, (train_index, valid_index) in enumerate(folds_index):
    df_train.loc[valid_index, 'fold'] = int(i)
    
# Convert to Spark DF
dfs_replication = spark.createDataFrame(df_train)
```

In the code below, Pandas UDF is defined. This function will be sent to each node to perform the training. If you are struggling to understand how Pandas UDF works, please read the post "[Leveraging Machine Learning Tasks with PySpark Pandas UDF](https://neowaylabs.github.io/data-science/Leveraging-Machine-Learning-Tasks-with-PySpark-Pandas-UDF/)".

Note that after training the model, it must be sent to cloud storage. These models will then be downloaded into the driver to make the necessary predictions. This was the way we found to transfer files from nodes to the cluster driver.


```python

# Declare the schema for the output of our function
outSchema = StructType(
    [StructField('fold', IntegerType(), True),
     StructField('Accuracy', FloatType(), True),
     StructField('model_path', StringType(), True),
     ])

# Decorate our function with pandas_udf decorator
@pandas_udf(outSchema, sf.PandasUDFType.GROUPED_MAP)
def run_model_by_fold(pdf):
    # 1. Get fold
    fold = pdf.fold.values[0]
    
    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(pdf.drop(['label', 'fold'], axis=1), 
                                                        pdf['label'], 
                                                        test_size=0.3, 
                                                        random_state=42)
    
    # 3. Create model using the pandas dataframe
    model_c = CatBoostClassifier()

    model_c.fit(X_train.values,
                y_train.values,
                eval_set=[(X_test, y_test)],
                plot=False,
                verbose=False)
    
    # 4. Evaluate the model
    accuracy = accuracy_score(model_c.predict(X_test), y_test)

    # 5. Save Model
    model_path = 'model' + str(int(fold))
    model_c.save_model(model_path)
    upload_file()

    # 6. Return results as pandas DF
    res = pd.DataFrame({'fold': fold,
                        'Accuracy': accuracy,
                        'model_path': model_path}, index=[0])
    return res
```

In this part, a groupby is performed on the "fold" column and a command is applied to send the Pandas UDF function "run_model_by_fold" to the nodes. Note that this Spark command is lazy.

```python
results = dfs_replication.groupby("fold").apply(run_model_by_fold)
```

Finally, a non-lazy function is executed and the models are trained. The total execution time is also calculated.

```python
import time
start = time.time()
results.sort('Accuracy', ascending=False).show()
end = time.time()
print(end - start)
```

    +--------------+----------+----------+
    |          fold|  Accuracy|model_path|
    +--------------+----------+----------+
    |             4|0.89330953|    model4|
    |             1|0.89264286|    model1|
    |             0|0.8911667 |    model0|
    |             3|0.88995236|    model3|
    |             2|0.88797617|    model2|
    +--------------+----------+----------+
    
    26.526451349258423

As a result, we have a total execution time of 26.5 seconds. There is also a table with the expected accuracy for each model. Why "expected"? Because this accuracy was generated by the validation set and not by the test set. This is exactly what will be done next. All models are downloaded and predictions are made on the test set. The test set will have a column for the predictions of each model. Let's see.

```python
centralized_models = [CatBoostClassifier() for i in range(N_FOLDS)]

for i, model in enumerate(centralized_models):
    # Download
    download_file()
    # Load
    model.load_model(f'model{i}')
    # Predict
    df_test[f'preds_model_{i}'] = model.predict_proba(df_test)[:,1]
    # Accuracy
    print(f"Model {i}'s accuracy: " + str(accuracy_score(df_test['label'].values, df_test[f'preds_model_{i}'].apply(np.rint).values)))
```
    Model 0's accuracy: 0.8922566666666667
    Model 1's accuracy: 0.8925666666666666
    Model 2's accuracy: 0.89263
    Model 3's accuracy: 0.89216
    Model 4's accuracy: 0.89311

Now we have the accuracy calculated across the test set. Note that they are very similar to each other and only fluctuate in the third decimal place. Note also that, despite coming from models trained on a sample, these accuracies are very close to the approaches that train on the entire dataset.

## Fourth approach: ensemble training

What if we were able to approximate even more these results (models that train on the entire dataset vs models that train on a sample)? This is possible through an ensemble of the results we obtained in the third approach. Next, the following ensembles will be held:

- Mean
- Median
- Maximum
- Stacking (random forest)


```python
models_cols = [f'preds_model_{i}' for i in range(0, N_FOLDS)]

df_test['pred_mean'] = df_test[models_cols].mean(axis=1)
df_test['class_mean'] = df_test['pred_mean'].apply(np.rint)

df_test['pred_median'] = df_test[models_cols].median(axis=1)
df_test['class_median'] = df_test['pred_median'].apply(np.rint)

df_test['pred_max'] = df_test[models_cols].max(axis=1)
df_test['class_max'] = df_test['pred_max'].apply(np.rint)
```

The centralized approach will be added here for benchmarking and comparison. 

```python
df_test['pred_single'] = centralized_preds
df_test['class_single'] = df_test['pred_single'].apply(np.rint)
```

In addition to comparing accuracies, let's also look at a Kernel Density Estimate (KDE) plot to get an idea of how well each model is separating the predictions. For more details, consult the reference itself [here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.kde.html).

Let's start by looking at our benchmark model, that is, our first approach. It is already expected that the centralized approach has the best accuracy, in this case, 0.9045. See from the KDE graph how the two classes are correctly separated from a threshold very close to 0.5.

```python
print(accuracy_score(df_test['label'].values, df_test['class_single'].values))

df_test.groupby('label')['pred_single'].plot.kde(legend='label')
```
    0.9045

<figure>
  <img src="/images/posts/distributed-training-using-pyspark/kde_centralized.png" alt="">
</figure>

The first ensemble to be observed will be the mean. Note that it brings an accuracy of 0.8969, that is, a difference of only 0.0076 compared to the first centralized approach.

```python
print(accuracy_score(df_test['label'].values, df_test['class_mean'].values))

df_test.groupby('label')['pred_mean'].plot.kde(legend='label')
```
    0.8969466666666667

<figure>
  <img src="/images/posts/distributed-training-using-pyspark/kde_mean.png" alt="">
</figure>

The second ensemble to be observed will be the median. Note that it brings an accuracy of 0.8963, that is, despite being lower than the mean, a difference of only 0.0082 compared to the first centralized approach.

```python
print(accuracy_score(df_test['label'].values, df_test['class_median'].values))

df_test.groupby('label')['pred_median'].plot.kde(legend='label')
```
    0.8963133333333333

<figure>
  <img src="/images/posts/distributed-training-using-pyspark/kde_median.png" alt="">
</figure>

The third ensemble to watch will be the maximum value. Note that it brings an accuracy of 0.8888, that is, the lowest among all ensembles, a difference of 0.0157 compared to the first centralized approach. It's not a bad result, but the other ensembles were better.

```python
print(accuracy_score(df_test['label'].values, df_test['class_max'].values))

df_test.groupby('label')['pred_max'].plot.kde(legend='label')
```
    0.88882

<figure>
  <img src="/images/posts/distributed-training-using-pyspark/kde_max.png" alt="">
</figure>


Finally, we are going to check our last ensemble approach through a stacking using the Random Forest model. The dataset will contain the predictions of each model plus the label.


```python
df_ensemble = df_test[[*models_cols, 'label']]
```

Just to get situated and let things make more sense, let's calculate the shape of the ensemble dataset. It has 300k rows, as expected, because our initial dataset has 1 million rows and we set the test set to have 30% of the data.

```python
df_ensemble.shape
```
    (300000, 6)

Once again, let's split the ensemble dataset into training and testing in the proportion 70%/30%.

```python
X, y = df_test[models_cols].values, df_test['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
```

Now, let's train the model and make predictions on the test set.

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=3, random_state=123)
clf.fit(X_train, y_train)

ensemble_preds = clf.predict(X_test)
```

The final accuracy is 0.8965, an intermediate result between the ensemble with the mean and the ensemble with the median. Overall, a great result. As the difference is small in relation to the ensemble with the mean, it is not possible to say which one is the best. Also because another model could have been used (instead of Random Forest) and even the same model could be tested with other hyperparameters. In our experiences, in some results, the stacking was better than the ensemble with the average.

```python
accuracy_score(y_test, ensemble_preds)
```
    0.8965333333333333



## Conclusion

This post presented three approaches to dealing with the "big data" problem, in addition to the centralized approach used as a benchmark.

The first centralized approach achieved an accuracy of 0.9045 and a time of 46.36s per fold. However, this approach is impractical when the amount of data is really large and was used here only as a benchmark for the other approaches.

The second approach achieved an accuracy of 0.9043 and a time of 256.21s per fold. That is, an accuracy practically equal to the centralized approach with a considerably longer training and prediction time given the decentralized nature of the solution.

The third approach trained several models centralized on different samples of the dataset reaching accuracies ranging from 0.89216 to 0.89311 in a time of 26.53s. That is, an accuracy close to the initial centralized approach with a shorter training and prediction time. It is important to note that the training was centralized, but was carried out in the nodes in a distributed way, reducing the time required for training.

The fourth approach showed how the third approach can come even closer to the first centralized approach. Through the ensemble of the different trained models, it is possible to achieve even greater accuracies.

It is important to remember that the accuracy of the third and fourth approaches tend to have a greater difference from the accuracy of the initial centralized approach when the dataset size decreases. But if that happens, the centralized training shown in the first approach becomes a real possibility.

Finally, there is no approach considered the best or ideal. It all depends on your scenario.

We hope you enjoyed the content. Let us know if you have any questions or suggestions.


*Authors: [Vitor Hugo Medeiros De Luca](https://www.linkedin.com/in/vitordeluca/), [Igor Siqueira Cortez](https://www.linkedin.com/in/igor-cortez-56793825/), [Luiz Felipe Manke](https://br.linkedin.com/in/luizmanke/)*
{: .notice}