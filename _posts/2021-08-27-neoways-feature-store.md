---
author: brenocosta
title: "Neoway's Feature Store"
categories:
  - data-science
tags:
  - pyspark
excerpt: "Feature Store"
---

A fundamental part of the process of building data science models is to create features that bring relevant information to the problem is being solved. This step is called feature engineering and it can be done by data science practitioners, be they data scientists, machine learning engineers, data analysts, or data engineers.

The feature engineering can be done in different ways. It can be just a selection of existing raw data, it can be an aggregation of fields after joining different tables, or it be a more complex function that is applied on the raw data to produce the input data that the model can interpret, and so on.

The data science practitioners write the feature creation source code in the git repository where the model is being developed. When it's deployed to production, the model pipeline has a step to run that code and generate the feature table containing all the required columns that the model is expecting as input.

This ad hoc approach - where features are created as needed by projects - may work for teams that are starting or small teams that only have a few models to maintain. However, this approach becomes inefficient when the organization scales.

The main problems caused by this approach were well explained in the book Machine Learning Design Patterns and they can be summarized below.

* Features aren't easily reused: features are created over and over again by individuals users or teams and never leave the project pipelines or notebooks in which they are created.

* Duplication of features: because of the lack of reuse, different projects created similar features duplicating the same business logic in the best scenario, but many times the same features are created differently.

* Poor data governance: there is no easy access in finding feature documentation to understand feature engineering details, to analyze dependencies, or to simply discover new features the can be used.

* Lack of consistency: divergence of the features in dev and prod, and challenges to deploy models in production. In many cases, data scientists need a different team to build the ETL jobs that will make the data available in production to run the model.

The Feature Store is emerging as a new kind of ML-specific data infrastructure to address those problems. The main idea is moving part of the feature engineering pipeline from the model repositories to a feature platform that is a centralized location to store and document datasets that will be used in data science models across projects and teams.

<figure class="align-center">
  <img src="/images/posts/neoways-feature-store/introduction_feature_store.jpg" alt="">
  <figcaption>Feature Store as a platform layer serving features to the machine learning models</figcaption>
</figure>

The decoupling provided by the feature store brings several benefits to the data science practitioners and to the company as a result.

* Consistency of features used in the development and production environment.

* Reduced time to create new models as data scientists have a catalog of datasets available at the start of development.

* Feature sharing between different teams increasing the contribution across the company projects.

* Creation of new curated data assets for the company that can be sold to customers.

## Software Architecture

Our Feature Store is fully integrated with the Neoway's data platform, making it easy tasks such as reading files from the data lake, registering schemas in the schemas API, producing data to Kafka, inserting records in Postgresql, indexing data in the Elastic Search and making them available to customers in the company's SaaS and APIs.

<figure class="align-center">
  <img src="/images/posts/neoways-feature-store/feature_store_architecture_simplified.jpg" alt="">
  <figcaption>Feature Store component is integrated with other data platform components</figcaption>
</figure>
