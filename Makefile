push:
	./yolo push --base "r8.im/stability-ai/sdxl@sha256:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316" \
	--dest r8.im/anotherjesse/sdxl-recur \
	--ast predict.py \
	--commit $(shell git rev-parse HEAD) \
	predict.py
