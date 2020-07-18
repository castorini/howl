#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# copy gitconfig from home
if [ ! -f ${DIR}/../.gitconfig ] && [ -f $HOME/.gitconfig ]; then
   cp $HOME/.gitconfig ${DIR}/../
   # install some default git config elements
   git config --file ${DIR}/../.gitconfig alias.lg "log --pretty=oneline --abbrev-commit --graph --decorate --all"
fi

# Variables required for logging as a user with the same id as the user running this script
export LOCAL_USER_NAME=$USER
export LOCAL_USER_ID=`id -u $USER`
export LOCAL_GROUP_ID=`id -g $USER`
export LOCAL_GROUP_NAME=`id -gn $USER`
DOCKER_USER_ARGS="--env LOCAL_USER_NAME --env LOCAL_USER_ID --env LOCAL_GROUP_ID --env LOCAL_GROUP_NAME"

# Variables for forwarding ssh agent into docker container
SSH_AUTH_ARGS=""
if [ ! -z $SSH_AUTH_SOCK ]; then
    DOCKER_SSH_AUTH_ARGS="-v $(dirname $SSH_AUTH_SOCK):$(dirname $SSH_AUTH_SOCK) -e SSH_AUTH_SOCK=$SSH_AUTH_SOCK"
fi

# Settings required for having nvidia GPU acceleration inside the docker
DOCKER_GPU_ARGS="--env DISPLAY --ipc=host --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw"

which nvidia-docker > /dev/null 2> /dev/null
HAS_NVIDIA_DOCKER=$?
if [ $HAS_NVIDIA_DOCKER -eq 0 ]; then
  DOCKER_COMMAND=nvidia-docker
  DOCKER_GPU_ARGS="$DOCKER_GPU_ARGS --env NVIDIA_VISIBLE_DEVICES=all --env NVIDIA_DRIVER_CAPABILITIES=all"
else
  #echo "Running without nvidia-docker, if you have an NVidia card you may need it"\
  #"to have GPU acceleration"
  DOCKER_COMMAND=docker
fi

IMAGE_NAME="ww4ff"
CONTAINER_NAME="ww4ff"
echo "Container name ${CONTAINER_NAME}"
if ! docker container ps | grep -q ${CONTAINER_NAME}; then
	$DOCKER_COMMAND rm ${CONTAINER_NAME}
	echo "Starting new container with name: ${CONTAINER_NAME}"
	$DOCKER_COMMAND run \
	$DOCKER_USER_ARGS \
	$DOCKER_GPU_ARGS \
	$DOCKER_SSH_AUTH_ARGS \
	-v "$DIR:/home/${USER}" \
	-v "/data:/data" \
	-it $ADDITIONAL_FLAGS --user root \
	--name ${CONTAINER_NAME} --workdir /home/$USER \
	--cap-add=SYS_PTRACE \
	--cap-add=SYS_NICE \
	--privileged \
	--net host \
	--device /dev/bus/usb \
	${IMAGE_NAME} \
	bash
else
	echo "Starting shell in running container"
	docker exec -it --workdir /home/${USER} --user $(whoami) ${CONTAINER_NAME}
fi

