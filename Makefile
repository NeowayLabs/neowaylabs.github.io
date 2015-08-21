.PHONY: build run clean

DOCKER_IMAGE = jekyll
WORK_DIR     = /srv/jekyll
CURRENT_PATH = $(shell pwd)
CURRENT_USER = $$USER
CURRENT_UID  = $$UID


build:
ifeq ("$(wildcard Dockerfile)","")
	sed "s@%USER%@$(CURRENT_USER)@g;s@%UID%@$(CURRENT_UID)@g" Dockerfile.template > Dockerfile
endif
	docker build --rm -t $(DOCKER_IMAGE) .

run: build
	docker run --rm -v $(CURRENT_PATH):$(WORK_DIR) -p 127.0.0.1:4000:4000 $(DOCKER_IMAGE)

clean:
	docker rmi $(DOCKER_IMAGE)

all: run
