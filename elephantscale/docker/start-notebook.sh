#!/bin/bash
set -e

# activate py35 environment
#source activate py35
#conda info -e

/usr/local/bin/start.sh  ~/run-bigdl.sh

# And run bash shell, so the container doesn't exit when Jupyter exits
# This way we can re-run ./run-bigdl.sh if needed to
/bin/bash
