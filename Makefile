UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_M),arm64)
	URL := "https://github.com/replicate/yolo/releases/latest/download/default.yolo_$(UNAME_S)_$(UNAME_M)"
else
	URL := "https://github.com/replicate/yolo/releases/latest/download/yolo_$(UNAME_S)_$(UNAME_M)"
endif


all: push

yolo:
	curl -o yolo -L $(URL)
	chmod +x yolo


push: yolo
	./yolo push --base "stability-ai/sdxl@sha256:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316" \
	--dest anotherjesse/sdxl-recur \
	--ast predict.py \
	--commit $(shell git rev-parse HEAD) \
	predict.py
