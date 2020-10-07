FROM debian:bullseye

# Locale stuff
RUN apt-get update && apt-get install locales -y
RUN sed --in-place '/en_US.UTF-8/s/^#//' /etc/locale.gen
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV JEKYLL_ENV dev

# Ruby + Jekyll dependencies
RUN apt-get update && \
    apt-get install \
    ruby-dev build-essential \
    patch zlib1g-dev liblzma-dev \
    libxml2-dev libxslt-dev  -y
RUN gem install bundler
RUN apt-get install git -y

# Application
WORKDIR /app
COPY Gemfile .
COPY Gemfile.lock .
COPY jekyll-theme-so-simple.gemspec .
RUN bundle install --path vendor/bundle
COPY . .
EXPOSE 4000
RUN gem update jekyll
ENV RUBYOPT=-W0
ENTRYPOINT ["bundle", "exec", "jekyll", "serve"]
