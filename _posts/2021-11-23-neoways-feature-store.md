---
author: brenocosta
title: "Neoway's Feature Store"
categories:
  - data-science
tags:
  - pyspark
excerpt: "Feature Store"
---

> This blog post was written with the support of talented people who developed Neoway's Feature Store together with me, [Manoel Vilela](https://www.linkedin.com/in/lerax/) and [Matheus Frata](https://www.linkedin.com/in/matheus-frata).

A fundamental part of the process of building data science models is to create features that bring relevant information to the problem being solved. This step is called feature engineering and it can be done by data science practitioners, be they data scientists, machine learning engineers, data analysts, or data engineers.

Feature engineering can be done in different ways. It can be just a selection of existing raw data, it can be an aggregation of fields after joining different tables, or it can be a more complex function that is applied to the raw data to produce the input data used by the model.

The data science practitioners write the feature creation source code in the git repository where the model is being developed. When it's deployed to production, the model pipeline has a step to run that code and generate the feature table containing all the required columns that the model is expecting as input.

This ad hoc approach - where features are created as needed by projects - may work for teams that are starting or small teams that only have a few models to maintain. However, this approach becomes inefficient when the organization scales.

The main problems caused by this approach were well explained in Lakshmanan's Machine Learning Design Patterns book and they can be summarized below.

* Lack of consistency: divergence of the features in dev and prod, and challenges to deploy models in production. In many cases, data scientists need a different team to build the ETL jobs that will make the data available in production to run the model.

* Features aren't easily reused: features are created over and over again by individual users or teams and never leave the project pipelines or notebooks in which they are created.

* Duplication of features: because of the lack of reuse, different projects create similar features duplicating the same business logic in the best scenario, but many times the same features are created differently and produce business errors that cause model performance errors.

* Poor data governance: when there is no easy access in finding feature documentation to understand feature engineering details and dependencies, or to simply discover new features that can be used.

In order to address those problems, the Feature Store is emerging as a new kind of ML-specific component. The main idea is to move part of the feature engineering pipeline from the model repositories to a centralized feature platform to store and document datasets that will be used in data science models across projects and teams.

<figure class="align-center">
  <img src="/images/posts/neoways-feature-store/introduction_feature_store.jpg" alt="">
  <figcaption>Feature Store as a platform layer serving features to the machine learning models</figcaption>
</figure>

When the feature engineering source code and pipeline are moved to a feature platform, there is decoupling from the model source code and pipeline, and several benefits to the data science practitioners and the company as a whole.

* Keep consistency of features used in the development and production environment, because there is no need to rewrite features. The data scientists (or data engineers) can write feature code during the development stage, and reuse the same code to generate features on the production pipelines.

* Increase feature sharing between different company projects and teams, since there is a centralized and curated catalog. This allows data scientists to get features previously created on successful data science models.

* Reduce time to create new models as data scientists have a catalog of feature datasets available from the beginning of the development stage. The models can quickly have a baselined version with the existing features.

* Improve data governance with a centralized and curated catalog, feature monitoring tools, access control, and permission management.

All of the above-mentioned benefits come with the cost of maintaining a new component in ML infrastructure. It's important to assess whether the return on investment in this component is worth it for your company's context.

## Software Architecture

In Neoway, we’ve developed an in-house Feature Store to make data scientists’ work easier during the model development either to discover datasets available for generating features or to create new feature datasets to be used on the models. [We have been developing tools for improving our feature pipelines since when the term ‘feature store’ was not popular](https://neowaylabs.github.io/data-science/neoways-feature-framework/) and that learning journey helped us in this new development.

Our Feature Store is fully integrated with the data platform, and this allows the interaction with data capabilities to perform tasks very easily, such as reading files from and writing files to the data lake, registering schemas in the Schema API, producing data to Kafka, inserting records in PostgreSQL, indexing data in the Elastic Search and making them available to customers in company's SaaS and APIs. The diagram below shows how that integration works in practice.

<figure class="align-center">
  <img src="/images/posts/neoways-feature-store/feature_store_architecture.jpg" alt="">
</figure>

As the image illustrates, there are some functions built on feature store to make interaction with other platform components easier:

* Register: create a new curated schema and register it on the catalog
* Read: use the catalog to read existing files from the data lake
* Ingest: write data frames into the data lake and make them available on the catalog
* Publish: produce a data frame previously ingested to the Kafka to make it available on the platform for consumers

The proposed architecture allows two kinds of feature serving: 1) offline serving that is used for batch training and prediction, the most common use case in our data science projects, and 2) online serving that is mainly used for stream prediction on model APIs. The next image presents the internal components of the architecture.

<figure class="align-center">
  <img src="/images/posts/neoways-feature-store/feature_store_architecture2.jpg" alt="">
</figure>

The main internal components are described below.

* Indexer: data lake indexer that uses rules configured using types and providers to scan buckets and to catalog files found in the feature store
* PostgreSQL: a database that contains information from the catalog, such as the registered schemas, properties and metadata, and the indexed file paths
* REST API: service that exposes some useful endpoints for the SDK to perform operations
* SDK: user-friendly Python SDK that allows you to perform operations in interactive environments or scheduled applications
* Transformers: service to automate the generation of features on production

This architecture can be easily extended to attend new use cases such as near real-time prediction using Kafka and Spark Streaming for generating stream features, but our business context still does not demand this type of feature serving.

## Schema Rules Everything

Every dataset available in the catalog needs to be registered on the schema registry using the company’s schema API. The registration step requires a schema identifier and a JSON schema.

### Schema Identifier

The schema identifier consists of three terms:

* schema type: data type according to the lifecycle (eg. entity, table, static, features, model)
* schema provider: team or client responsible to produce data (eg. neoway, analytics, solution delivery, customer data science)
* schema name: meaningful name to identify data (eg. company_estimated_revenue, person_income_prediction, company_compliance_risk)

Let's see some examples of datasets in the Feature Store identified by type, provider, and name.

<figure>
  <img src="/images/posts/neoways-feature-store/schema_identifier_1.png" alt="">
</figure>

The first dataset identifier example has a type `table`. It comes from a relational database, a provider `neoway` because it’s produced by the company’s default ingestion flow, and a name `company` because it contains data about companies.

<figure>
  <img src="/images/posts/neoways-feature-store/schema_identifier_2.png" alt="">
</figure>

The second dataset identifier example has a type `model` because it’s an output produced by data science model, a provider `analytics` because it’s provided by the Analytics team, and a name `company_compliance_score` because it contains the compliance score model outputs for the companies.

### JSON Schema

The schemas must be registered using a valid [JSON schema](https://json-schema.org/) that contains schema properties (name, type, and description of each one), required properties, keys, description, and other custom attributes. JSON Schema is a vocabulary that allows you to annotate and validate JSON documents, and it's a powerful tool for validating the structure of JSON data. We’re using it to validate backward compatibility.

The first version of the schema just needs to be valid, but when a schema is updated it has to be backward compatible to ensure that its dependencies are not broken. The topics below should be guaranteed for the second and next versions of the schema.

* The field types need to be the same in the old and new schema versions
* After setting a required field, the default value for it needs to be filled
* After setting a default value, the field always need some value as default
* The fields can not be removed
* Just one schema per array field will be validated

The schema validation guarantees a contract between applications and prevents people or applications from breaking it.

## Python SDK

One of the main advantages when using the Feature Store is to discover new datasets that are provided by different teams across the company that can be useful to develop to create features or models.

The SDK helps the data discovery by providing a curated data catalog easy to use and a text-based search. In addition, it also allows visualizing dataset details such as metadata, schema, and file versions. All this is in your favorite Python environment. Let's explore the catalog.

### Catalog and Search

The catalog gives an overview of the data available on the Feature Store. It's organized in hierarchy levels to improve navigation. Each hierarchical level has an auto-complete feature implemented to use on interactive environments like REPL or Jupyter Notebook.

Browsing the catalog is interesting but it can take time to find an interesting dataset or find some properties that might be useful. The search allows us to find datasets or properties quickly because it performs a full-text search on the datasets using the term you search for, and it will return a table of datasets containing that term.

You can view the details of a dataset after finding some data that is useful to use in your project. Once an interesting dataset has been found, it's possible to view the dataset details by using the catalog entry. The animation below shows an example of catalog usage.

![catalog search](/images/posts/neoways-feature-store/fs_catalog_search.gif)

### Read

You can use the SDK to read dataset versions from the catalog. The function `read()` uses spark data frame API for reading data lake files to data frames. After you can use the [spark API](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.html#pyspark.sql.DataFrame) to manipulate your data according to your need. The example below presents the reading of the company_compliance_score model's latest version from the data lake. 

![catalog read](/images/posts/neoways-feature-store/fs_catalog_read.gif)

### Register and Ingest

Data transformation is the process to convert data from a format to another one. The most common transformation is to convert raw data into curated data, remove duplicated data, convert data types, and enrich data. More complex transformations include joins between different tables, aggregations to generate grouped data, and so on.

Let's see an example of how to use a dataset to make a simple transformation and create a new useful data frame using the dataset of company partners.

![register ingest](/images/posts/neoways-feature-store/fs_register_ingest.gif)

The schema of the transformed data frame is used to generate the JSON schema and to register on Neoway's schema registry. The schema is available in the catalog soon after being registered.

Once the schema is registered, you can ingest data frames into the Feature Store. You can use the method `fs.ingest()` to ingest a data frame into the feature store. This method is performed in three steps: 1) write data frame to the data lake, 2) insert into feature store database, and 3) update catalog instance.

### Publish

A data frame that was previously ingested can be published to the Kafka, and data will be available for consumers such as ElasticSearch, MongoDB, Neo4j, Big Query, and so on.

![publish](/images/posts/neoways-feature-store/fs_publish.gif)

## Transformers

Feature Store Transformers is a project to centralize the code of our curated transformations. Today, our data scientists use to register their feature "recipes".
However, the project supports any kind of transformation, that's why the name
Transformers! It's responsible for automating the execution of feature engineering pipelines on production.

### How to add a new transformation

The data scientists just need to open a merge request on the project's git repository. The branch must contain a new feature folder with three files: python module with feature code, JSON file with schema configuration, and python module with expectations module.

Transformers follows this file structure:

```
transformers/
  SCHEMA_TYPE/
    SCHEMA_PROVIDER/
      SCHEMA_NAME/
        __init__.py
        code.py
        config.json
        expectations.py
  ...
  <other internal modules>
```

#### code.py

The feature code is written as a python module with a standard function `def transform(catalog: Catalog) -> DataFrame:` that is called during the pipeline execution. The function receives a catalog instance as input and returns a spark data frame as output. The developer is free to read datasets from the catalog and make the feature transformations. Follows a very simple transformation example:

```python
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from feature_store.catalog import Catalog


def transform(catalog: Catalog) -> DataFrame:
    df = catalog.table.neoway.company_partners.read()

    return (
        df
        .groupby("cnpj")
        .agg(F.count("partner_code").alias("partnersCount"))
    )
```

Note the developer doesn't need to know anything about infrastructure provisioning.

#### config.json

The JSON configuration file contains fields that are used by the register method. The developer must inform schema identifier (schema type, provider, and name), as schema fields (description, key, field descriptions).

```json
{
    "schema_type": "features",
    "schema_provider": "analytics",
    "schema_name": "company-compliance-score",
    "description": "Number of company partners",
    "key": ["cnpj"],
    "field_descriptions": {
        "cnpj": "Company code",
        "partnersCount": "Partners count"
    }
}
```

#### expectations.py

The expectations file is a python module with data quality tests written by developers. This file holds quality tests for the transformation. What is an Expectation? It's an assertion that you can make the data, verifying if it passes its expected value. You can create one by calling human-readable functions on a helper class that we made available for development.

```python
from transformers.quality import ExpectationsDataset


def expect(df: ExpectationsDataset):
    # cnpj
    df.expect_column_values_to_be_unique("cnpj")
    df.expect_column_values_to_not_be_null("cnpj")
    df.expect_column_value_lengths_to_equal("cnpj", value=14)

    # partnersCount
    df.expect_column_min_to_be_between("partnersCount", min_value=0, max_value=100)
```

We've moved from an approach using unit tests to a new approach using data tests to make it easier to add features without putting the quality aside. Using the traditional software engineering unit tests for feature code discourages data scientists from using the tool to contribute with new features. For this purpose, we've adopted [Great Expectations](https://greatexpectations.io/) to run the quality pipeline successfully.

### CI/CD pipeline 

The project uses a CI/CD pipeline for running the features pipelines on three different branch types: feature, develop, and master branches. The pipeline includes default tasks to build, lint, and test project code, but it also includes feature pipeline steps such as running expectations test, registering schemas, ingesting files to the data lake, publishing datasets to Kafka.

|         | build | lint | tests | transform | expectations | register | ingest | publish |
|---------|-------|------|-------|-----------|--------------|----------|--------|---------|
| feature |   x   |   x  |   x   |     x     |              |          |        |         |
| develop |   x   |   x  |   x   |     x     | x            |  dry-run |        |         |
| master  |   x   |   x  |   x   |     x     | x            |     x    |    x   |    x    |

The [Airflow](https://airflow.apache.org/) is being used to schedule and run the pipeline on production. Both CI/CD runner and Airflow are responsible to launch spark jobs to a Kubernetes cluster, and this makes our pipeline highly scalable.

## Conclusion

Neoway's Feature Store has brought several benefits to the company's teams. The curated catalog has provided good data governance for the datasets, increasing data quality and improving security. Python SDK has simplified many complex tasks like interacting with other platform components, and this reduces time to develop data science models. Code rewriting has been reduced between development and production environments because professionals are using the same tools for both. 

One of the consequences of those benefits is that we're seeing other teams outside the data science domain wanting to use the Python SDK to be more efficient to deliver solutions for the customers. On the other hand, some customers are realizing value in the features created in the Feature Store and paying for them to use in their data science models.
