## pick a specific version, to ensure predictability
FROM jupyter/pyspark-notebook@sha256:64420e4c348ab48fb806f42332109cbc205ae74cda67240c3e0974c5f7e6e969
## or latest
#FROM jupyter/pyspark-notebook@latest

MAINTAINER Elephant Scale <info@elephantscale.com>

## --- CONFIG
ARG MAVEN_VERSION=3.5.2
ARG SCALA_VERSION=2.11.8
ARG SPARK_VERSION=2.2.0
ARG SBT_VERSION=1.0.2
ARG INSTALL_DIR=/usr/local
ENV BIGDL_HOME  ${INSTALL_DIR}/BigDL

#ARG BIGDL_URL=https://s3.amazonaws.com/elephantscale-public/BigDL/BigDL.zip
## Download BigDL from release page
## https://bigdl-project.github.io/0.3.0/#release-download/
ARG BIGDL_URL=https://repo1.maven.org/maven2/com/intel/analytics/bigdl/dist-spark-${SPARK_VERSION}-scala-2.11.8-linux64/0.3.0/dist-spark-${SPARK_VERSION}-scala-2.11.8-linux64-0.3.0-dist.zip
## --- end CONFIG

USER root

## apt update
RUN apt-get update -yq && \
    apt-get -yq dist-upgrade

## basic utils + jdk
RUN apt-get install -yq  --no-install-recommends \
    atop \
    curl \
    less \
    openjdk-8-jdk-headless \
    rsync \
    unzip \
    wget \
    zip

## install maven
RUN curl -fsL  http://apache.cs.utah.edu/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz | tar xfz - -C ${INSTALL_DIR}

RUN cd ${INSTALL_DIR} && rm -f maven && ln -s apache-maven-${MAVEN_VERSION}  maven

ENV MAVEN_HOME ${INSTALL_DIR}/maven
ENV PATH=$MAVEN_HOME/bin:$PATH


## Scala expects this file
RUN touch /usr/lib/jvm/java-8-openjdk-amd64/release

## Install Scala
RUN \
  curl -fsL https://downloads.typesafe.com/scala/${SCALA_VERSION}/scala-${SCALA_VERSION}.tgz | tar xfz - -C ${INSTALL_DIR}

RUN cd ${INSTALL_DIR} && rm -f scala && ln -s scala-${SCALA_VERSION}  scala

ENV PATH=${INSTALL_DIR}/scala/bin:$PATH

## Install sbt
RUN \
  curl -L -o sbt-${SBT_VERSION}.deb "https://dl.bintray.com/sbt/debian/sbt-${SBT_VERSION}.deb" && \
  dpkg -i sbt-${SBT_VERSION}.deb && \
  rm sbt-${SBT_VERSION}.deb && \
  apt-get update && \
  apt-get install sbt && \
  sbt sbtVersion

## install spark
RUN \
  curl -fsL "https://d3kbcqa49mib13.cloudfront.net/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz" | tar xfz - -C ${INSTALL_DIR} && \
  cd ${INSTALL_DIR} &&  rm -f spark && ln -s spark-${SPARK_VERSION}-bin-hadoop2.7  spark


## cleanup apt
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

## ----- install BigDL
RUN \
    mkdir -p ${BIGDL_HOME} && \
    cd ${BIGDL_HOME} && \
    wget --quiet "${BIGDL_URL}"  && \
    unzip *.zip && \
    rm -f *.zip

## disable sudo password
RUN echo "${NB_USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

ENV PATH=${INSTALL_DIR}/spark/bin:$PATH

# this is where volumes will be mounted
RUN mkdir /work


### now as a regular user
USER $NB_USER

## update conda
# RUN conda update --all
## install NLTK
RUN  conda  install -y  nltk

## python 3.5 env
RUN conda create -y -n py35 python=3.5 numpy scipy pandas scikit-learn matplotlib seaborn jupyter nltk tensorflow opencv pillow

## python 2.7 env
RUN conda create -y -n py27 python=2.7 numpy scipy pandas scikit-learn matplotlib seaborn jupyter nltk tensorflow opencv pillow

# list envs
RUN conda info -e

## install tensorboard with pip  for py27
RUN /bin/bash -c "source activate py27 && \
    pip install tensorboard && \
    source deactivate"

## install tensorboard with pip for py35
RUN /bin/bash -c "source activate py35 && \
    pip install  tensorboard && \
    source deactivate"

## source python 3.5
RUN echo "source activate py35" >> ~/.bashrc

## source python 2.7
## This will be the default python env
RUN echo "source activate py27" >> ~/.bashrc

# working directory
ENV WORKING_DIR ${HOME}/work
RUN rm -rf ${WORKING_DIR} && ln -s /work  ${WORKING_DIR}

## --- copy files last, so not to bust the cache ---

COPY run-bigdl.sh  $HOME/

USER root
RUN mv /usr/local/bin/start-notebook.sh   /usr/local/bin/start-notebook-old.sh
COPY  start-notebook.sh  /usr/local/bin/
RUN chmod +x /usr/local/bin/start-notebook.sh


## finally switch back to jovyan to avoid accidental container runs as root
USER $NB_USER
