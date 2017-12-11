# How to use the Docker image

## Building

```bash
    $   cd  bigdl-tutorials/docker

    # to use default 'Dockerfile'
    # do not forget this DOT (.) at the end
    $   docker build  .

    #   to do a different build file
    # do not forget this DOT (.) at the end
    $   docker build  -f  Dockerfile-v3   .

    # to add a tag
    $   docker build  -f  Dockerfile-v3  -t bigdl/bigdl   .
```

To force a build provide  `--no-cache ` option
```bash
    $   docker build --no-cache   .
```

To see built images

```bash
    $   docker images
    $   docker image inspect  <IMAGE_ID>
```

## Running it

```
$   docker images
```
Will list all images.  

#### Option 1:
The simplest way to run the docker container is to use `run-bigdl-docker.sh` script at the project root directory.  It will run the image and also mounts a working directory so all the work is saved automatically.

```
    $    ./run-bigdl-docker.sh   <image_id>
```

#### Option 2:
Running the Jupyter notebook by default  (~/run_jupyter.sh)
```
    $   docker run -it  -p 8888:8888   IMAGE_ID
```

#### Option 3:
Shell access
```
    $   docker run -it  -p 8888:8888   IMAGE_ID   bash
```

In Docker container you can run Jupyter as follows
```
    $    ./run_jupyter.sh
```

Go to http://localhost:8888  in browser


## Pushing to Dockerhub

```bash

    ## login
    $  docker login
    ## enter username, password

    ## tag the image
    $   docker tag IMAGE_ID  elephantscale/bigdl-sandbox:latest
    $   docker images

    ## Pushing
    $   docker  push  elephantscale/bigdl-sandbox:latest

```
