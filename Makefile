DOCKER_IMG = neowaylabs

all: serve

install:
	@if [ ! -d vendor ]; then \
         bundle install --path vendor/bundle; \
	     bundle add jekyll; \
    fi


serve: install
	bundle exec jekyll serve

clean:
	rm -rfv _site vendor


docker-run:
	docker run -it --rm --network=host $(DOCKER_IMG)

docker-build:
	docker build -t $(DOCKER_IMG) .
