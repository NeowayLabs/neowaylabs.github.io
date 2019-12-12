---
author: yurivs
title: "DataOpscience? Learning vs Doing"
image:
  path: /images/posts/dataopscience/experiment.jpg
  thumbnail: /images/posts/dataopscience/experiment.jpg
categories:
  - programming
tags:
  - dataops
  - software-development
excerpt: "Pragmatically exploring and choosing tools for better Analytics"
---

To work with the same tool everyday for years can make you bored, even miserable. Moreover, being stuck to it probably means having to tinker to make it do things it is not suitable for. At the same time, a company with no standards whatsoever has very little chance to not fall apart from lack of knowledge dissemination and toolbox reusability. So how to balance it out?

Our Data Scientists mostly work with Python (pyspark, pandas), and SQL to build up models. Those are not enough to solve our problems: models still need to be deployed, data needs to be ingested, preprocessed and made available to them, then scores and predictions must be indexed on our platform so that our clients can use the results. As some colleagues point it: "that smells like DataOps to me". And these summarize quite well some of our activities, which could be further described by the idea of providing a frictionless ecosystem, bringing abstractions, and sometimes even indirections, to processes, and let people do their own job as well as they can, worrying as little as possible with technicalities.

# Meanwhile, in our Universe

If that seemed too vague for you, it’s because it is not a closed problem. As seen from [a previous post](https://neowaylabs.github.io/programming/from-notebooks-to-production-software/), turning a jupyter notebook exploration into a real world product is not a trivial task. Even with standards for packaging models into artifacts, we still need to provide a way for those routines to be called/executed, safely access data they need, and then route results into appropriate sinks. And then we have the exceptions, which must have tailor-made solutions. At times, that can be (and oh boy it is) challenging. On the other hand, quite exciting too! On my own view, I am glad to have the opportunity to not only be exposed to, but actually learn and build things on 5 languages and a bunch of other tools, and that's not even a year since I joined the team!

To be fair, some of those things aforementioned are not more than proof of concept applications, which are a great part of our day to day, experiment and understand the right tool for the job. For example, using Helm charts to spin up Apache CouchDB clusters, configure native Erlang query server, and populate it with all sorts of data and routines, and conclude that, as U2, we still haven't found what we are looking for. Some may complain that's a waste of time or a lot of work for no result at all, but here I stand to point out that knowing why something isn't a good fit is just as important as why it would be. With the understanding that some tools solve our problems, and others may not, no matter how hyped, well designed and built they might be, we focus on the problem at hand. Therefore, we get to deeply analyze and comprehend issues and necessities, themselves, which is more valuable than expertise on any tool.

# So it's just playing around?

No! We have much more than only ~~failed~~ experiments. This very year, we added a Scala (Spark on our Kubernetes cluster) job to gather metadata from all our sources. We also translated a Clojure application into Rust (one that dealt with integration with other teams’ platforms), adding further error handling, fallback strategies, backoffs, reduced network transit and memory footprint (therefore costs). As of now, experiments are being conducted on some streaming and integration frameworks on the Java ecosystem (also Scala and Kotlin), looking for ways to bring more reactive behaviours to our Analytics team, taking the leap from _Big_ to _Fast_ Data.

Maybe the most consolidated tool would be Apache Airflow. For those unfamiliar with it, there we can use bundled and third-party operators, designed to perform tasks into the world, ranging from local bash/python scripts to direct Google Cloud Platform services, like Dataproc and Storage. You can chain these operators into Direct Acyclic Graphs (DAG), describing dependencies, branching, watchers and triggers, and with these building blocks, we implement our pipelines, not so different from the one bellow:

<figure class="align-center">
  <img src="/images/posts/dataopscience/example-dag.png" alt="">
</figure>

In case you are wondering, that's how to structure it:

{% highlight python %}
etls = [gen_etl(f"domain_{i}") for i in range(1, 4)]
fins = [finalizers(f"domain_{i}") for i in range(1, 4)]

[check_new_data, get_credetials] >> start_cluster

for (ingest, normalize_aggregate) in etls:
    (start_cluster
     >> ingest
     >> normalize_aggregate
     >> retrain_on_new_data)

(retrain_on_new_data
 >> predict 
 >> [stop_cluster, check_integrity]
 >> start_sender)

for (index, persist) in fins:
    (start_sender
     >> persist
     >> index
     >> stop_sender)
{% endhighlight %}

As one can see, just models don't add up to a complete data analytics pipeline. Other tasks are needed, such as getting the right dataset(s) (which can come from multiple places), intermediary persistence, sharing results to other teams or publishing it to the client platform. This is better done outside the model logic, so as to decouple us from specific infrastructure details.

# How are we doing?

Is that too much? We are a 2 person squad (for now, stay tuned: we are looking forward to bump those numbers on 2020), and most of the needs from the dozen Data Scientists and Machine Learning Engineers in our team get to be covered in a reasonable time, while keeping exploration and innovation alive, however, we could use some extra hand to take more projects out of the realm of ideas. So for us, technological diversity seems to be going hand in hand with our continuous improvements.
