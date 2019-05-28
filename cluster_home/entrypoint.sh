#!/bin/bash
# This is executing in the container.
set -e

source $HOME/.bashrc

export TMP=/tmp/grasping
export TMPDIR=/tmp/grasping
export TEMP=/tmp/grasping

mkdir -p /tmp/grasping

exec $@
