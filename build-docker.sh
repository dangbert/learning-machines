#!/bin/bash
set -e

IMAGE="cigroup/learning-machines:local"
docker image build . --tag "$IMAGE"
