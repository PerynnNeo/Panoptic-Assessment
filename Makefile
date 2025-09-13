EDGE_IMAGE=edge-processor:latest
EDGE_NAME=edge_processor

build:
\tdocker build -t $(EDGE_IMAGE) ./edge

run-file:
\tdocker run --rm --name $(EDGE_NAME) \\
\t  --cpus=4 --memory=8g \\
\t  -e VIDEO_SOURCE="samples/input.mp4" \\
\t  -e SAMPLER_MODE=motion -e MOTION_THR=12.0 -e HEARTBEAT_S=2.0 \\
\t  -v $$PWD/results:/results \\
\t  $(EDGE_IMAGE)
