DOCKER_IMG = neowaylabs

all: serve

install-deps:
	@if [ ! -d vendor ]; then \
         bundle install --path vendor/bundle; \
	     bundle add jekyll; \
    fi

blog: image
	docker run -it --rm --network=host $(DOCKER_IMG)

image:
	docker build -t $(DOCKER_IMG) .

serve: install-deps
	bundle exec jekyll serve

clean:
	rm -rfv _site vendor

