# neowaylabs.github.io

Neoway Tech-Blog.

# Posts

To write a new post, create a file in markdown inside of `_posts` dir.
For projects pages, put the file on `_projects`.

A template of header for post it is something like this:

```
---
author: super_programmer
title: "Why NLP is so Awesome?"
categories:
  - programming
tags:
  - nlp
  - machine-learning
excerpt:  "A report about the process of applying NLP to Neoway models"
---
```

But you are not a author yet? Create your profile [here][authors] too and make a [Pull Request][pr]!

[authors]: https://github.com/NeowayLabs/neowaylabs.github.io/blob/master/_data/authors.yml
[pr]: https://help.github.com/en/articles/about-pull-requests

# Running

Requirements:

* Docker

After installing Docker just run:

```
make image
make blog
```

And the blog should be available at 127.0.0.1:4000.

# Running on host

Requirements:

* Ruby
* Ruby-bundler
* GNU Make

After installing the dependencies just run:

``` bash
make serve
```

The blog should be available at 127.0.0.1:4000.
