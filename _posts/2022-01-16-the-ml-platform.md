---
author: brenocosta
title: "The journey of creating and evolving our machine learning platform"
categories:
  - machine-learning
tags:
  - feature-store
  - feature-engineering
  - pyspark
  - python
  - machine-learning
  - data-science
---

[Neoway](www.neoway.com.br) is a Brazilian company that generates intelligence for decision-making and productivity for the processes of marketing, compliance, fraud prevention, legal analysis and credit management. Since its foundation, the company has invested on a data platform for collection, ingestion, treatment and presentation that can be used to create products and services for different use cases. 

In the last five years, data science and machine learning have emerged to bring more intelligence in solving customer problems, and currently play a key role in the solutions that make up the portfolio. Models, indicators, intelligent journeys are examples of data science-powered applications delivered on our web application or via APIs.

The company's growth brought new challenges to maintain a good communication structure among teams and fast business value delivery to the customers.

## Organizational Structure

For a long time the company was organized by functional areas like platform, data, data science, application, product. This kind of organization caused functional silos that led to problems such as lack of systemic thinking, and hard dependency management during software development.

Data science team struggled to get historical data with the format required for modeling tasks. The team also had difficulties deploying model predictions integrated with the company's architecture. Both tasks had dependencies with internal teams and the services were not provided properly.

That organizational structure resulted in inefficiency to deliver business value to our customers. At the end of 2019 the company redesigned the teams into cross-functional ones capable of delivering end-to-end solutions and accelerating the software delivery.

### On Team Topologies

One of the theoretical references used to redesign the teams was the Team Topologies book. It brings new ideas around effective team structure for enterprise software delivery. 

The book authors reinforce the importance of [Conway's law](https://www.melconway.com/Home/Conways_Law.html){:target="_blank"} for effective team organization. That theory states that “organizations which design systems are constrained to produce designs which are copies of the communication structures of these organizations”. 

The speed of delivery is heavily affected by the organization’s design and the dependencies among its teams. A fast flow of execution requires restricting team communication as much as possible. 

Team Topologies book proposes to apply a kind of reverse engineering of Conway’s law by **designing teams to match the desired architecture**! This is team-first thinking. It suggests a business oriented team design for fast flow based on concepts such as cognitive load, fundamental team types and interaction modes.

### Our Team Redesign

Our team redesign was performed by separating business domains and creating teams to maximize the software delivery to our customers. The functional teams were changed to cross-functional teams named **cells**. They are small units of delivery that work collaboratively.

The **stream-aligned** and **platform** are the most fundamental team types in our company. Stream-aligned teams have a continuous flow of work aligned to a business domain. They need to be empowered to deliver customer value as quickly and independently as possible. Platform teams provide internal services to reduce the complexity for the stream-aligned teams during software development and delivery.

The image below presents a very summarized view of some cells with which our machine learning platform team interacts the most.

<figure>
  <img src="/images/posts/ml-platform/cells_overview.png" alt="">
</figure>

The product and service cells are responsible for delivering value directly to the end customers. They need to be empowered to build and deliver customer value as quickly and independently as possible. Those cells are created to better serve the business domain requirements.

On the other hand, the platform cells have two main responsibilities. The first is to simplify the use of complex technologies that require a high level of specialization, such as kubernetes, spark clusters, distributed systems, data lake, streaming platform, iam, security, data warehouses, search engines. The list doesn't finish here.

The second responsibility is to prevent business cells from starting to build multiple solutions for solving the same problems. When a common problem arises for several people, platform cells act to create a single solution that solves the problem more efficiently thinking in the long term.

Cells have three interaction modes: **collaboration**, **facilitating**, and **x-as-a-service**. Collaboration is when teams work together on a part of the system. Facilitating is when a team acts as a mentor on a specific topic to help another team. X-as-a-service is when a team provides and another team consumes something as a service such as a self-service API for example.

The new working model has brought many benefits:

* Delimiting responsibilities brings more clarity to teams and helps dependency resolution. 
* Communication between related cells improves idea exchange  and increases the software reusability. 
* The business value delivery flow becomes clearer and faster!

## Internal Platform Teams

[Platform teams](https://martinfowler.com/articles/platform-prerequisites.html){:target="_blank"} aim to reduce the cognitive load of stream-aligned teams by providing services used by software developers or data scientists that are generally a combination of internally developed and/or externally acquired features.

A platform team analyzes the internal teams’ requirements, looks for external solutions, and makes them more convenient for developers to use on a daily basis. In this context, one of the main characteristics of a good platform is the ease of use and it’s extremely desirable that services are offered in self-service mode with well-written documentation and good support for developers.

Suppose a data scientist with a background in statistics, economics, or physics was hired to create models that bring a high business return to the company. Your scope of work should not include knowledge of details on how to provision infrastructure for deploying models. The platform team would act by solving the inherent complexities and providing a simplified user experience.

The discussion about the importance of platform teams is not something new, but the decision to create it must be considered according to each context. The decision to create an in-house platform team should be a balance between the long-term efficiency and quality benefits versus the financial costs involved in building and evolving it.

Neoway has in its DNA the strengthening of the platform teams, considering the high gains obtained in the creation of different solutions that generate value to customers.

### Data Platform

Since having good data is a requirement for creating a good model, the Data Platform plays a central role in the data science context. Some of Neoway's main data platform components are briefly described below.

* Schema registry: service with a publicly exposed API that allows teams and applications to register new data schemas. Every data transferred on the platform must be registered with a valid JSON Schema. The schema changes must maintain backward compatibility, which prevents the crash of applications running in production.

* Data Hub: enables data traffic within the company using the Kafka streaming platform as the central bus. Data producers can register schemas and produce data to Kafka topics and make them available to everyone. Consumer applications can consume data from the Data Hub when necessary.

* Data Lake: makes data available in Google Cloud Storage to enable applications that need to read/write data in batches. It’s very useful for using ephemeral spark clusters for processing massive amounts of data.

* Data Warehouse: provides the data in BigQuery, a tool that allows the storage and consultation of large data sets. Teams can use it to run SQL queries and generate quick insights in a scalable cloud environment. Business Intelligence applications can use it as a storage and processing engine.

* Data API: makes data available for consumption via REST API, either by internal applications or by external customers. One of Neoway's business models is to sell data produced internally on the platform to its consumers via API.

The machine learning platform is built on top of the data platform components. Our team is both a consumer and contributor of the internal data platform components.

### ML Platform

The machine learning platform has components for the major stages of the model delivery lifecycle: data exploration, feature engineering, model development, model deployment and monitoring. 

Our solutions are mainly used by data scientists working in stream-aligned cells who need to do tasks such as provisioning infrastructure for running distributed spark kernels, accessing the data lake in an easy and organized way, creating and running model feature engineering jobs, deploying models into production, ensuring that the models run with quality.

The image below presents a summary of the components of our platform and the integration with other existing components in the company.

<figure>
  <img src="/images/posts/ml-platform/ml_overview.png" alt="">
</figure>

Data pipelines are created to prepare the raw data that is produced by other applications in a structured format to be used by data science tasks. Every dataset schema is registered and it’s available to users in an organized way for easy discovery and reading.

The Feature Store facilitates the creation and consumption of features used in data science models. It maintains consistency between development and production by using the same feature code on both environments. It encourages reusability of features across projects and teams. We’ve published an entire blog post on our Feature Store. [Check it out for more details](https://neowaylabs.github.io/data-science/neoways-feature-store/){:target="_blank"}.

Development Environment enables data scientists provisioning the infrastructure of distributed spark clusters with a set of tools installed and configured in an easy way. The users can do any kind of data science tasks like data exploration, feature engineering, model training at scale!

Model productization template and tools include a data science template to package the source code. It has batteries included to create python modules, run unit and integration tests on CI pipeline, publish docker images in the registry, write and run dags in Airflow.

Pipeline orchestration is the part responsible for running batch pipelines for model training and scoring. We use Airflow to create the DAGs that run periodically, but we use the concept of DAG as a configuration. The data scientist does not need to write complex operators. Just create a file and fill in some configuration parameters. Then just play and run the model to test and if everything is working, the model goes into production. 

Model APIs can be used to serve models. They are used in some very specific use cases. Binary artifacts from the models created in the productization step are loaded during initialization and used for making predictions by key.  This type of strategy is mostly used by the professional services team according to the clients' needs. We also have on-demand pipelines where training and prediction pipelines are triggered by the users.

## Structuring the ML Platform Team

At the beginning we had three main challenges to overcome as a machine learning platform team.

**Integration with existing solutions in the company**: we needed to improve integration with different services of the company provided mainly by the other platform teams. 

At that time, teams were not thinking of providing their services for the other internal teams, so the user experience was not good and documentation was not sufficient for understanding how to configure and use those services without the support of other people.

**Developer experience of our software components**: when we worked on the data science team, there were some projects developed for data preparation, feature engineering, model deployment, etc. 

The projects were not developed taking into account aspects such as user experience and a systemic view of the components. The result over time was a bad experience and low user adoption of some components.

**MLOPs challenge**: we were facing some problems at that time with the models running in production. How to make a good governance of features and models? How to deploy models in an efficient way? Many open questions to be answered. 

A dedicated team with mixed skills in software engineering, infrastructure, and machine learning could stay focused on how to propose solutions to the main problems brought up by data scientists. While, data scientists are more focused on business and statistical skills.

### Product Thinking

The first decision was how to approach the challenges above mentioned in that moment. 

A technical team may be tempted to solve the problems only from a technical perspective by considering aspects such as scalability, reliability, and maintainability. Those things are important, but the platform solutions can’t fail at a crucial point: user adoption.

In the blog post [Mind the platform execution gap](https://martinfowler.com/articles/platform-prerequisites.html){:target="_blank"}, the authors have suggested to prioritize developer experience, and it’s one of the aspects to mind the platform execution gap. The reason is simple: a product that no one uses is not a successful product, no matter its technical merits.

Our team decided to approach the platform with product thinking. We decided to focus on the user needs and experience, but deliver solutions with technical excellence. 

From the beginning, we interviewed the users, tried to understand their journey, and listened to their pain points. It's necessary to see any user perspective on the platform components being developed. An advantage of internal products is that we have closer contact with users through the company’s communication channels. **User empathy is the key!** 

The next sections present some tools we use to improve our understanding of the machine learning platform users.

#### User Interviews

Our main mission in this stage was to create a connection with people and hear what they had to say about the tools available at that time. Our interview form had three basic questions:

1. How do you currently do your data science related tasks?
2. What are your current difficulties?
3. What are your suggestions for improvement?

These basic questions can generate so much insight and backlog for the team!

We talked in some user interviews about the [features framework](https://neowaylabs.github.io/data-science/neoways-feature-framework/){:target="_blank"}, our first set of tools for feature engineering. We found that the user experience was terrible, and that was the main reason why our feature creation tool stopped being used. Those insights helped us a lot on [the development of our feature store solution](https://neowaylabs.github.io/data-science/neoways-feature-store/){:target="_blank"}.

#### Proto-Personas

We used a simple description of proto-personas to describe who are the current and future users of our platform. The image below presents a brief description of some characteristics to clarify it.

<figure>
  <img src="/images/posts/ml-platform/product_proto_personas.png" alt="">
</figure>

Proto-personas give an insight into the characteristics of users, which guide the team to make good decisions in an empathetic way during the software design and development.

#### User Story Map

The user interviews generated user stories to describe the user needs in a structured way. We used a [user story map](https://www.nngroup.com/articles/user-story-mapping/){:target="_blank"} for creating the entire view of the data science lifecycle. 

User story map defines 3 types of actions: 

* activities that represent the high-level tasks that users aim to complete in the product; 
* tasks that represent the steps in a sequential order to achieve the activity and; 
* stories that provide a more detailed level of user needs. 

The image below shows an example of our initial user story map.

<figure>
  <img src="/images/posts/ml-platform/product_user_story_map.png" alt="">
</figure>

One of the benefits after [making a user story map](https://www.oreilly.com/library/view/user-story-mapping/9781491904893/){:target="_blank"} is to have a systematic understanding of the user journey in a single page.

#### Prioritization

After talking and understanding the user needs, the team tried to understand the most urgent priorities to improve the users' workflow and the effort required to deliver them. 

Since we’re starting many projects, it was virtually impossible to make reasonable estimates for any user story. The team could not yet learn from its own track record. In this scenario, senior engineers can help to provide an effort estimation at a macro level. 

The next step was to group user stories into development epics to solve specific problems (model deployment, feature engineering, etc.). The [prioritization matrix](https://www.productplan.com/glossary/2x2-prioritization-matrix/) was used as a natural framework to help the decision making on what’s the most important epics to be developed comparing effort vs value.

<!-- <figure>
  <img src="/images/posts/ml-platform/product_prioritization_matrix.png" alt="">
</figure> -->

Another important aspect of prioritization is considering the company’s objectives in the medium and long term. Strategic and tactical managers can help more clarity about that. After getting input from users and stakeholders, we created OKRs to provide a way to measure our achievements. An OKR can be something like this: “reduce time to put model into production from X to Y“ and the development epics must be related with one OKR.

#### Execution

We decided to use [scrum](https://en.wikipedia.org/wiki/Scrum_(software_development)){:target="_blank"} as a framework for developing and delivering our solutions. Our sprint planning generally starts on a Monday and we discuss what’s the best strategy to generate more value for our users by prioritizing the right things. 

Our sprint backlog is composed of development epics previously planned on OKRs, operation, and support tasks. It’s common to have some problem someone faced in the previous week and we need to fix it in the new sprint, or someone needing support to deliver some project. 

After starting the sprint, we have daily meetings with the entire team to assess progress and resolve any blocks that happen. This meeting is important to keep everyone's commitment to what was agreed. Seeing the progress of the tasks on the board renews the motivation to stay focused and seeing the lack of progress turns on the alert for any problems that might be happening.

Our software engineering culture is mature. Every piece of software needs to be reviewed by team members and only can be merged after being accepted by at least one person. Automated tests, documentation, changelog are very important for us, and you use Gitlab for automating CI/CD pipelines. The tasks are done during a biweekly sprint, and we finish with a sprint review and retrospective.

## Key Takeaways

During the journey of creating and evolving the machine learning platform at Neoway, we’ve faced a lot of challenges but it has been an awesome learning experience for the team. Some takeaways during this time:

Team design takes an influence on the company software architecture and fast flow.

* Our organizational structure changed for the better, team scopes are more clear and communication between teams is designed for fast flow.
* Don’t design teams too small, try to optimize for autonomy, and set communication types when teams need to work together.
* A X-as-a-service communication type needs good services and documentation.


Approaching a machine learning platform with product thinking helps to understand the users and generate more value as a lean team

* Understanding your users is the most important thing you have to do. [Do things that don’t scale](http://paulgraham.com/ds.html){:target="_blank"}!
* Involve users from the beginning, show them prototypes, let them use the product, get feedback and improve from there. Don’t involve everyone, prefer more mature and curious users with an early adopter profile.
* A small team needs to optimize for fast and small deliveries. 


Internal teams trust internal platform services if they deliver value, are easy to use, and are reliable.

* You will need the understanding of your users when the services present any problem. Don’t let your users down. Provide good support and you will gain their trust.
* Use every interaction with your users to get feedback and improve the services. Payment comes in the long term.
