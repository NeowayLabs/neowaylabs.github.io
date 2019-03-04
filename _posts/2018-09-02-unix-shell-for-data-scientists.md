---
author: boechat107
title: "7 Unix Commands Every Data Scientist Should Know"
description: "7 Unix Commands Every Data Scientist Should Know"
categories: programming
tags:
  - shell
  - unix
---

In many situations, unprepared data scientists can spend much more time than
necessary on secondary tasks.
Although their attention should stay on analyzing data, checking hypothesis,
engineering features, etc., they often need to get their hands dirty and code
auxiliary scripts and parsers to get the information they need.

Prepared data scientists prefer to use Unix commands and solve their secondary
tasks in a few seconds.

This post presents 7 basic Unix commands (maybe eight) that, once incorporated
in the day-by-day toolset, can potentially improve the productivity of any data
scientist.
The beauty of these tools is the ability to easily combine them into powerful
commands, executing tasks that could take a full day of coding to get them
right.

This is the selected list, probably in the order of most frequent usage:

* [grep](#grep)
* [cat](#cat)
* [find](#find)
* [head/tail](#headtail)
* [wc](#wc)
* [awk](#awk)
* [shuf](#shuf)

In addition, it is shown how two auxiliary commands (`xargs` and `man`) can
improve even further the usability of the 7 commands above.

# 7 Unix Commands

## grep

`grep` searches for patterns in files or in the standard input. It is really
useful to find stuff without opening files in an editor.


### Examples

**Problem:** We want to find where the library **ggplot** was used in any script
in the directory **r_project**.


```
$ grep -rn ggplot r_project/
r_project/script.R:4:library(ggplot2)
r_project/script.R:247:t<-ggplot(base.test.c, aes(x=score, colour=gap_dic, group=gap_dic))
r_project/script.R:265:t<-ggplot(base.test.c[base.test.c$gap_total<40000,], aes(x=gap_total, colour=corte2, group=corte2))
```
* option `-r` stands for recursion
* option `-n` tells `grep` to show the line numbers

**Problem:** From a big log file, we want to get only the logging messages with
the pattern `TRAINING -`.

```
$ grep 'TRAINING -' my_app.log
```

Note that `grep` is case sensitive by default.

**Problem:** Given a directory composed of subdirectories whose names contain a
date, we want to get the complete name of one or more subdirectories containing
a date like `2018_05`.

```
$ ls models_by_date/ | grep 2018_05
```

Here we use the pipe `|` operator to send the output of the first command to
`grep` by the standard input [^1].

[^1]: The standard input can be thought as a special kind of file whose contents commands have access to. The pipe operator is able to take the standard output of one command (that would otherwise be printed on the screen) and send it to the standard input of another.

**Problem:** When listing all installed Python packages, we want to see only the
results containing the name **gpyopt** (usually a single line).


```
$ pip freeze | grep -i gpyopt
GPyOpt==1.2.5
```

`pip freeze` returns a list of installed packages to the standard output and
`grep` searches for **gpyopt** (`-i` makes it case insensitive) in the standard
input.

**Problem:** Given an arbitrary text file, we want to show only those lines
containing a pattern specified by a *regex*.

```
$ grep -oP "'[\w]+ == [\d.]+'" python_library/setup.py
'numpy == 1.15.0'
'fire == 0.1.3'
'gpyopt == 1.2.5'
'recsys_commons == 0.1.0'
```

This example uses Perl regular expression to search for packages and versions in
a Python setup file.


## cat

Prints on the screen (or to the standard output) the contents of files. Simple like
that.

```
$ cat script.sh
#!/bin/bash

set -eo pipefail

echo "$BLEU"
```

## find

`find` searches for files by specifying many different (and optionally) kinds of
parameters. It is also able to execute some simple actions or an entire command
line using the resultant files.

### Examples

**Problem:** We want to find all files with the extension **json** in the
current directory, including subdirectories.

```
$ find . -name '*.json'
./third-party/wiwinwlh/src/26-data-formats/example.json
./third-party/wiwinwlh/src/26-data-formats/crew.json
```

**Problem:** All files with extension **pyc** must be removed from the directory
`my_library/modules`, recursively.

```
$ find my_library/modules -name '*.pyc' -delete
```

**Problem:** In a directory containing multiple projects, we want to find all
`setup.py` files that contain the text `boto3` (in other words, we are looking
for projects using the library **boto3**).


```
$ find . -name setup.py -type f -exec grep -Hn boto3 {} \;
```

Here we tell it to search only for files with the exact name **setup.py**,
ignoring directories (`-type f`) and executing `grep` (`-exec`) on each one of
them (why don't we just use `grep` alone?)[^2].
Note that:
* `{}` is used to pass a file path as an argument to `grep`
* `\;` marks the end of the command
* `grep` parameter `-H` makes it print the filename

[^2]: Using `grep` recursively would bring all lines of all files containing `boto3`; the result might be too big to be useful.

## head/tail

Like working with *DataFrames*, `head` and `tail` print the first and the last
lines of files or of the standard input.

### Examples

**Problem:** Given a CSV file, we want to quickly look at its header.

```
$ head -n 1 data.csv
```

**Problem:** From a potentially huge log file, we want to read only its last
20 events.

```
$ tail -n 20 app.log
```

## wc

`wc` is very useful to count lines, words or even characters in files.

### Examples

**Problem:** How many text lines does a file have?

```
$ wc -l data.csv
624 data.csv
```

**Problem:** We want to know the total number of CSV records in a directory
containing multiple CSV files.

```
$ wc -l data_dir/*.csv
102224 data_dir/part-00000-02aa95cd-3907-44c8-87ee-97ff44677349-c000.csv
102513 data_dir/part-00001-02aa95cd-3907-44c8-87ee-97ff44677349-c000.csv
204737 total
```

We may need to discount the number of header lines (this could also be done just
using the shell).

## awk

`awk` uses a programming language (AWK) for text processing. It is powerful and
might seem complicated to learn and use. However, there are a few commands that
can be used very frequently.

### Examples

**Problem:** Given a CSV file, we want to know the number of columns just by
analyzing its header.

```
$ head -n 1 data.csv | awk -F ',' '{print NF}'
91
```

First, `head` sends the first line to the standard output, which is consumed (as
the standard input) by `awk` and broken up into a sequence of fields delimited
by `,`. `NF` holds the number of fields in a line.

**Problem:** Given a big CSV file containing hundreds of columns, we want to
have a quick look at the first lines of a specific column (let's say the third
column).

```
$ head data.csv | awk -F ',' '{print $3}'
var_x
3.0
4.0
3.0
3.0
3.0
3.0
3.0
2.0
3.0
```


## shuf

`shuf` generates a random permutation of its inputs. It is very useful to sample
data.

### Examples

**Problem:** Given a CSV file, we want to take a random sample (50 records) from
it and save this sample in another file.

```
$ cat big_csv.csv | shuf | head -n 50 > sample_from_big_csv.csv
```

The operator `>` is used to redirect the standard output to a normal file.
Without using it, the result of `head` would be just printed on the screen.

**Problem:** Given a directory containing multiple data files, we want to get a
random sample of files (5 files) and copy these files to another directory.

```
$ find origin_dir/ -type f | shuf | head -n 5 | xargs -i cp {} sample_dir/
```

First `find` returns a list of files in `origin_dir` (including this directory
name in their paths), then `shuf` shuffles the list of file paths, `head` takes
the first 5 file paths and, finally, `cp` copy each of these 5 files to the
directory `sample_dir` (`xargs`, explained in the next section, is used as an
auxiliary command since we couldn't just use the standard input).


# Auxiliary Commands

## xargs

`xargs` is a kind of auxiliary program, since its role is to convert the
standard input into an argument of another program. This is really useful to
make a chain of processing programs that don't use the standard input.

### Examples

**Problem:** We must remove all **__pycache__** directories in a given project
directory.

```
$ find my_app -name '__pycache__' -type d | xargs -i rm -r {}
```

As `find` action `-delete` can't remove nonempty directories, we use `rm -r` to
remove all `__pycache__` folders.

**Problem:** Remove all Git branches that were already fully merged with their
upstream branches.

```
$ git branch | xargs -i git branch -d {}
```

## man

`man` is also another auxiliary program. It provides an interface to reference
manuals of almost all UNIX commands. The image below shows the result of
checking the manual of `man` itself (`man man`).


![unix_manual](https://upload.wikimedia.org/wikipedia/commons/d/db/Unix_manual.png)
> By Kamran Mackey - Arch Linux using the Cinnamon display
> manager.,GPL.

# Conclusion

This post presented 7 Unix commands to increase the productivity of data
scientists, including examples of usage and two auxiliary commands to make even
more powerful combinations.
By describing common problems and possible solutions, the intention is to make
the post a useful reference to tackle similar scenarios.

More time we spend exploring and using these commands, more productive we become
by using them.
