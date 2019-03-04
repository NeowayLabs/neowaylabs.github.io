all: serve

install:
	@if [[ ! -d vendor ]]; then \
         bundle install --path vendor/bundle; \
	     bundle add jekyll; \
    fi


serve: install
	bundle exec jekyll serve

clean:
	rm -rfv _site vendor
