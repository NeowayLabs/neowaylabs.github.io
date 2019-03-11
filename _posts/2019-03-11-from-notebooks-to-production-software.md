---
author: gbenatt92
title: "From Notebooks to Production Software"
categories:
  - programming
tags:
  - data_science
  - software_development
excerpt:  "Personal experience on learning software development as a 'data scientist'."
---

![how it feels to deploy software](/images/posts/doge_production.png)

*how it feels to deploy software*


## Some personal context

In the beginning I thought I knew how to code. I knew some OOP and a good amount of Python (although nothing impressive) and I could get things done with a rudimentary glue code. It was never anything amazing, but it worked and I could train models, perform experiments and predict some data as it came in.

Most of my work was prototyped locally in a Jupyter Notebook and I never thought much of how of it I should put into  production, if there was reproducibility, if code review was ensured and so on. And at other times the main bulk of the code remained in the notebook and was never made into a proper project until told otherwise.

Then I had my first experience putting a Machine Learning model I've made into production and it was quite a ride! The code review was awesome because the senior developers really took their time and pointed out every single inconsistency, mistake, formatting issues and even architectural flaws. I believed that for them going through the messy “notebook turned into project” was more than nightmarishly, but for me their feedback and code review was eye opening.

I guess this was the turning point which pushed me try to learn how to develop software instead of just learning how to code.

The initial course of action was to just adapt every single architectural mistake I've ever made, such as calling a bunch of ad-hoc queries into the SQL database or loading all the data into onto the RAM... Regularly. . Oops. Of course this worked well with training a sample, but I could never scale up to deploy without a lot of ugly and frail solutions.

About 2 years ago I felt the weight of maintaining software in production. It actually amazed me because I had to start revising, refactoring, adding features and, to top it off, I was constantly seeing my poor code crumble like a house of cards.

I delved deeper into the basics of software development and started programming with a deliberate curiosity. I would constantly ask for code reviews, even though it wasn't part of my team’s culture, and make myself available to look into other people's code (although I could never contribute much). Most importantly, I learned how to write and structure software in a simple and pleasant way to read.

So, my journey has led me to this point, from someone who thought they knew how to develop software to someone that kinda knows a bit. Now I'll discuss a bit more about the lessons I wish I would’ve learned sooner, in the context of data science.

## Reproducibility is key

Warning: If it works on your machine only, it's the same as not working at all. Do you want to wake up every morning and run a series of scripts to make sure the model is up to date? What if you decide to leave the company? What if you have so many different models running at the same time?

This is way too common in Data Science, especially because a lot of professionals are self-taught in programming and most companies have no standards for delivering a Machine Learning project.

I can't stress it enough, because I've made this mistake countless times, if all you have is lose scripts, notebooks or a code base without any instructions to run through, it's the same, if not worse, as not delivering the project.

Why? Because, if you can't reproduce the results then there’s no assurance that  these are correct and that they are reliable enough to make decisions on that could potentially cost your company and/or it's clients much more than your paycheck.

Although prototyping is fine and dandy during the early stages, there should be an urge to move on all of your ideas and code into a reproducible project with clear instructions on how to run it and including tests where it's required.

## No one is an island

A big reason on why I see a lot of frustrated, burnt-out Data Scientists is because either they’re  in an environment with little collaboration and no peer review culture or being in teams where they are the only ones programming or doing a analytical job.

There's also, in some cases, this horrible culture where they feel there's no time to do code review, that they can't afford time to learn proper software skills and that feedback on your work is only there to hurt you and to make you lose your deadline.

Here I have a few examples of what Data Scientists can gain when they collaborate and constantly expose their work to peers:

* You learn insight into new things and improve yourself and your craft;
* The quality of the code tends to improve;
* Some subtle but grave mistakes (such as data leakage or bad sampling) can be caught early on;
* You avoid being the *only* one with the knowledge about the project;
* The team as a whole improves.

This lone-ranger cowboy culture might be the biggest thing that I hope to see gone. It was created through years of media and companies glorifying Data Scientists, whereas it contributed into making the Data Science scene seem amateurish at best.

If you are the only Data Scientist in your company, you can seek out other software engineers, colleagues or stakeholders with analytical skills for feedback on results, solution architecture and so on. Having a feedback loop is really important for personal growth and improvement.

## Always seek to move forward

I deeply believe that a good Data Scientist should be, at least, a good analyst and a good software developer. You can't forsake one in favor of the other because you need to know how to improve both and constantly try to add to your craft.

You need to always improve your analytical capabilities, fresh up your intuition and challenge yourself to understand the behaviours that your data manipulation and analysis uncover. You even need to go a step further and revise old conclusions you once thought were true.

But your models, projects and analysis will often times need to be in production. There are no shortcuts and rarely you'll find places where there'll be anyone besides yourself that will refactor your project and deploy it. Seek and listen to the advice from senior Software Engineers, always ask for code reviews and avoid forsaking quality at all costs.

Something I've realized is that even when developing ETLs and quick queries, it was all still software, it did not live in a vacuum, it did cost me time and energy and it often delivered value or had side-effects. I'd be fooling myself if I didn't give thought to my work and jumped to the next project without constantly trying to improve.

Although, initially, I did slow down my pace to refactor my code, revise architectural mistakes or even to learn style guides for the language I was using, it paid over in orders of magnitude later. Even prototyping and experimenting on notebooks are more effective now than ever before.

## Some final thoughts

![me at work](/images/posts/doge_programmer.gif)

This was a small round-up of my 5-ish years of trying to be a Data Scientist.

Writing this up made me see that a lot of things here are more related to learning software development than anything else and, in fact, it was and is my biggest challenge.

All throughout this time, there was never a day I didn't feel fortunate to work with people that were way more competent than I was, so I have my peers to thank for any progress I've made.

