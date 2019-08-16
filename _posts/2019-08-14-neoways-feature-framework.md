---
author: mfrata
title: "Neoway's Feature Framework"
image:
  path: /images/posts/neoways-feature-framework/blog_poster.png
  thumbnail: /images/posts/neoways-feature-framework/blog_poster.png
categories:
  - data-science
tags:
  - feature-engineering
  - pyspark
excerpt: "Scalling data science development"
---

As a company grows it becomes more difficult to maintain all of its data science projects. The way the features are generated across these projects can introduce some problems such as the following: duplicates, lack of features consistency, absence of documentation, or even scalability issues. Have you ever heard of two teams using the same features on different problems while being calculated differently? Or a data scientist having to discard a whole Feature Engineering procedure due to the cumbersomeness to adapt to other projects?

The problems listed above are commonplace in analytics teams today. And they cannot be ignored if your company wants to scale. Still, how do we tackle these issues? Well, there isn’t a silver bullet solution. But, here at [Neoway](http://neoway.com.br/), we had great results by building a **Features Framework**.

It is a set of applications that store, generate, ensure quality, and serve the features for our stakeholders. They may be members of the data science team in Florianópolis, a single data scientist at the New York office or even a client who wants to make use of our features on their models.

In this blog post, we will dive deep into the internals of the **Neoway’s Features Framework**, explaining all of its components and discussing the outcomes for our company.

# Where the features live

There are more than 700 entities that are processed periodically and are stored on Neoway’s data lake. The entities that are going to be used by the framework’s calculation are converted to the parquet file format for better operability, space usage, and querying efficiency. The generated features go back to the data lake in a parquet file format.


<figure class="align-center">
  <img src="/images/posts/neoways-feature-framework/features_store.png" alt="">
  <figcaption>The Features Store is a subset of our data lake.</figcaption>
</figure>


  The framework is cloud-agnostic but it currently requires provisioning of cluster management and other supporting tools for scheduling and serving. These tools, generally, are offered by a cloud provider. They can simplify the task of infrastructure handling.

# The main engine

The main component of the framework is the **Features Builder**. It is an extensible framework for features development, where they are logically organized by domains with a specific primary key.

<figure class="align-center">
  <img src="/images/posts/neoways-feature-framework/features_builder.png" alt="">
  <figcaption>Features Builder and two domains flow.</figcaption>
</figure>


The Features Builder is built with [PySpark](http://spark.apache.org/docs/latest/api/python/index.html) since it provides a nice trade-off between convenience and scalability. In other words, we can deal with a high volume of data because of Apache Spark while being easy to our Data Scientists to create new features since Python is a language with great readability and low barrier to entry.

To illustrate this process further, we will code two features for a given business context which belongs to a specific domain. For instance, we wish to build a feature for credit score (business context) of a company (domain). Thus, we create a new module and function with unit tests and documentation:

{% highlight python %}
# business_context1.py
import datetime

import pyspark.sql.functions as sf
from pyspark.sql.column import Column
from pyspark.sql import DataFrame

def feature_one(entities, date):
    """
    feature_one: doc string here
    """
    df = entities["entitiy1"].select("col1", "col2", ..., "colN")
    return df.WithColumn("feature_one", _new_feature_calculation(...))

def feature_two(entities, date):
    ...

# test_business_context1.py
import pytest

from featuresbuilder.business_context1 import feature_one

def test_scenario_with_feature_one(mocks, sparksession_fixture, ...):
    ...
    assert expected == feature_one(entities, date)

def test_sceneario_with_feature_two(sparksession_fixture, ...):
    ...
{% endhighlight %}

  Then a merge request is opened and the reviewers check for two aspects of the new features: the Software and the Feature Engineering aspect. The former is to make sure that all good practices are being followed such as code structure, if the tests are well suited, code scalability and so on. The latter looks for business rules and the calculation of the feature itself.

  Another advantage of this way of feature development is something that we call: **Backward Feature Generation**. It is a primordial "feature" of the Features Framework (pun intended). Whenever a new feature comes, its calculation not only occurs on the current timestamp but also for past dates whenever historical data are available. Using the same example described above, if our `feature_one` is created in October but the data source exists since January, the feature will be extended to those dates. In essence, this means that `feature_one` is available for multiple dates starting from January, even if its conception only happens in October. This capability is great as it allows us to backtest our models.


# Can we make serving easier?

The real value of the Builder and the Store only comes when our stakeholders are making use of the features. However, accessing a storage bucket on the cloud is not the best way to achieve that. To solve this, we developed a high-level layer to access those: the **Features Selector**.

<figure class="align-center">
  <img src="/images/posts/neoways-feature-framework/features_selector.png" alt="">
  <figcaption>Data access for statistical analysis and modeling.</figcaption>
</figure>


The **Selector** is a Python package that can be easily imported on the data scientists' notebooks. It provides a catalog to view history, files, schema, features list and more importantly: to get a complete dataset of features, including their history, to use for  statistical analysis and modeling. We automated the serving of clusters and integrated that into our modeling workbench template (more about it on a future blog post), allowing our Data Scientist to work with Big Data. Therefore, Pyspark is key library to the **Selector**. The code snippet below shows just how easy it is to generate a training dataset using it.


{% highlight python %}
>>> from featureselector import Selector

>>> domain_features = Selector("domain", sparksession)

>>> domain_features.catalog()
>>> # ['t', 't-1', 't-2' ,..., 't-N']

>>> domain_features.schema()
>>> # {'feature_one': type, "feature_two": type, ...}

>>> # User passes timestamps and features
>>> df = (domain_features
...      .get_features('t', 't-1')
...      .select('pkey', 'feature_one', ..., 'feature_N'))
{% endhighlight %}


# How about turning it into a product?

At first, the Features Framework was aimed to solve our internal scalability challenges. As it grew we saw the potential that it could bring also to our clients. We proved our hypothesis using the Selector to deliver the features to our customers. Nonetheless, someone still had to stop and do the job of generating them. Meaning that it would not scale well. That led to our external way of serving it: the **Features API**.

<figure class="align-center">
  <img src="/images/posts/neoways-feature-framework/features_api.png" alt="">
  <figcaption>Data access for customers (RESTfull).</figcaption>
</figure>


The API is a service written in Golang that provides a low latency interface for features consumption and also has built-in security and billing capabilities provided by our Platform team. A key aspect is that we only include features that we use and have a proven value in real world applications. As it turns out, those features are also bringing results to our clients in many segments such as finance, insurance, recovery, retail, consumer goods and many others. Before requesting the features, the client can access its metadata as well as the timestamps available.


# Let’s talk about data quality

Until now we saw all the value of the Features Framework can create. But how do we ensure quality? You’ve guessed it: **Features Monitor**.

<figure class="align-center">
  <img src="/images/posts/neoways-feature-framework/features_monitor.png" alt="">
  <figcaption>Monitor checks for statistics of our features.</figcaption>
</figure>


The Monitor runs after the Builder’s calculation is complete. It has two main objectives: to check for simple quality **descriptive statistics** and the **diff statistics** related to the previous timestamps.

The descriptive statistics is delivered in a json format with summary information such as null index, number of rows, distribution, mean, standard deviation and others depending on the type of the feature. Moreover, it also includes some warnings to highlight dreadful variables. The warnings are triggered  when an attribute is all null or NaN, if it has low variability, or it is constant.

The diff statistics looks for changes in the schema, and the rate of change of the features’ distributions triggering a warning if it surpasses a certain threshold. This mechanism enables our team to audit this feature and trace the root cause of the change in the data.

{% highlight python %}
{
  "overview": {
    "domain": "domain1",
    "number_of_features": 0000,
    "pk": "domain_pk",
    "features": [
    {
      "name": "feature_one",
      "type": "feature_one_type",
      "number_of_rows": 0000,
      "percent_nullNaN": 0.00000
      "summary" : {
        "stat1": ...
        "stat2": ...
      }
    },
    {
      "name":"feature_two",
      ...
    },
    ...
    ]
  }
}
{% endhighlight %}

<figure class="align-center">
  <figcaption>Example of Features Monitor report.</figcaption>
</figure>


# Le grand orchestrateur

All of the components above are crucial parts of the framework. Yet there is another hidden component that plays a big role in the features’ calculations. We’re talking about the orchestrator, the scheduler. We’re talking about [Apache Airflow](https://airflow.apache.org/).

<figure class="align-center">
  <img src="/images/posts/neoways-feature-framework/features_framework.png" alt="">
  <figcaption>All the components of the Features Framework.</figcaption>
</figure>


Airflow is a platform that is used to schedule and monitor workflows. You can code a directed acyclic graph (DAG) which executes a set of tasks determined as desired. These tasks can vary from a simple bash command, a python code abstract in a container, a connection with the cloud and etc. And all of them can be done using Operators whose main responsibility is to provide abstractions to build DAGs with ease.

Our framework makes use of some operators: Docker, Cloud access, and bash. And the tasks vary from launching and killing clusters, executing the builder for the business contexts, triggering the Monitor, to saving the results in the Data Lake.

# Conclusion

We had a tour of our **Features Framework**. Starting at how the features are stored, passing through the calculation engine (builder), then learning about the serving methods and quality assurance process. Finally, we had a grasped on Airflow and how it keeps the project running periodically.

But what are the take-outs of this project? Firstly, the standardization and reuse of features. Being able to share them across the teams and projects in a way that is easy to operate. Secondly, the consistency between development and production environment, centralizing at the features at the Store that are used during experimentation and production. Moreover, the reduction of the project duration by 60 percent, since, a considerable amount of time is typically spent on Feature Engineering. Lastly, we could extend these benefits to our customers leading to an increase in our revenue, transforming internal solutions into technological products.

So should you invest in your Features Framework at your company? It depends on the stage you are at. But if you wish to scale up it is a good idea to invest in a solution to improve the feature development. And the Features Framework has a great return on investment. What do you think about our Features Framework? Leave us a message below or drop us a line.


*Special thanks to all people involved on the development of this framework: [Andre Boechat](https://www.linkedin.com/in/boechat107/), [Breno Costa](https://www.linkedin.com/in/breno-c-costa/), [Gabriel Alvim](https://www.linkedin.com/in/gabriel-benatti-alvim-520b69a2/), [Igor Cortez](https://www.linkedin.com/in/igor-cortez-56793825/), [Luis Pelison](https://www.linkedin.com/in/luis-felipe-pelison-243791107/), and [Manoel Vilela](https://www.linkedin.com/in/lerax/).*
{: .notice}
