#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

if [ "$1" = 'jekyll' ]; then
    if [ ! -e "/srv/jekyll/index.html" ]; then
        cd /srv && jekyll new --force jekyll
    fi
    cd /srv/jekyll && jekyll serve
fi
