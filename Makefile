DOCKER_IMG = neowaylabs
DOCKER_VOLUME_ARGS = -v ${PWD}/images:/app/images -v ${PWD}/_posts:/app/_posts/

all: serve

install-deps:
	@if [ ! -d vendor ]; then \
         bundle install --path vendor/bundle; \
	     bundle add jekyll; \
    fi

blog: 
	docker run -it --rm --network=host -e JEKYLL_ENV='dev' $(DOCKER_VOLUME_ARGS) $(DOCKER_IMG)

image:
	docker build -t $(DOCKER_IMG) .

serve: install-deps
	bundle exec jekyll serve

clean:
	rm -rfv _site vendor

