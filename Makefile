push:
	./yolo push --base "r8.im/stability-ai/sdxl@sha256:1bfb924045802467cf8869d96b231a12e6aa994abfe37e337c63a4e49a8c6c41" \
	--dest r8.im/anotherjesse/asim \
	--ast predict.py \
	--commit $(shell git rev-parse HEAD) \
	predict.py
