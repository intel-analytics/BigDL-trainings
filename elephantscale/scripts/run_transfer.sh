#!/bin/bash

## Usage
#   BIGDL_HOME=~/apps/BigDL   SPARK_HOME=~/apps/spark    ./run-bigdl.sh

export PYSPARK_PYTHON=$(which python)
export BIGDL_HOME=/usr/local/BigDL

# Check environment variables
if [ -z "${BIGDL_HOME}" ]; then
    echo "Please set BIGDL_HOME environment variable"
    exit 1
fi

if [ -z "${SPARK_HOME}" ]; then
    echo "Please set SPARK_HOME environment variable"
    exit 1
fi

# activate py35 environment
#source activate py35
# activate py27 environment
source activate py27
conda info -e


#setup paths
export BIGDL_JAR_NAME=`ls ${BIGDL_HOME}/lib/ | grep jar-with-dependencies.jar`
export BIGDL_JAR="${BIGDL_HOME}/lib/$BIGDL_JAR_NAME"
export BIGDL_PY_ZIP_NAME=`ls ${BIGDL_HOME}/lib/ | grep python-api.zip`
export BIGDL_PY_ZIP="${BIGDL_HOME}/lib/$BIGDL_PY_ZIP_NAME"
export BIGDL_CONF=${BIGDL_HOME}/conf/spark-bigdl.conf

# Check files
if [ ! -f ${BIGDL_CONF} ]; then
    echo "Cannot find ${BIGDL_CONF}"
    exit 1
fi

if [ ! -f ${BIGDL_PY_ZIP} ]; then
    echo ${BIGDL_PY_ZIP}
    echo "Cannot find ${BIGDL_PY_ZIP}"
    exit 1
fi

if [ ! -f $BIGDL_JAR ]; then
    echo "Cannot find $BIGDL_JAR"
    exit 1
fi

#cd work/notebooks/inception_v1

pwd

${SPARK_HOME}/bin/spark-submit  \
  --master local[4] \
  --driver-memory 16g \
  --properties-file ${BIGDL_CONF} \
  --py-files ${BIGDL_PY_ZIP} \
  --jars ${BIGDL_JAR} \
  --conf spark.driver.extraClassPath=${BIGDL_JAR} \
  --conf spark.executor.extraClassPath=${BIGDL_JAR} \
  --conf spark.sql.catalogImplementation='in-memory' \
  ./inception_transfer.py

