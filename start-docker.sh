#!/bin/bash
set -e

#IMAGE="cigroup/learning-machines:python3"
IMAGE="cigroup/learning-machines:local"

PROJECT_FOLDER="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )/"
echo "PROJECT_FOLDER=$PROJECT_FOLDER"
exec docker run --rm -it -v "${PROJECT_FOLDER}:/root/projects/" --net=host "$IMAGE" bash
