#!/bin/bash

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd -P )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd -P )"

# Default execution
SVE='python "'$DIR'/suppnet.py"'

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh . # Very dirty solution but works, by:
# https://github.com/conda/conda/issues/7980

conda activate suppnet-env

echo Calling $SVE "$@"
# which python
eval $SVE "$@"

conda deactivate
