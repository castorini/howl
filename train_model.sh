#bin/bash
set -e

ENV_FILE_PATH=${1} # env file which contains training settings
MODEL_TYPE=${2} # model type to use (e.g. res8)
WORKSPACE_PATH=${3} # workspace path where the trained model will be stored

if [ $# -lt 4 ]; then
  echo 1>&2 "invalid arguments: ./train_model.sh <env file path> <model type> <workspace path> <dataset1> <dataset2> ..."
  exit 2
fi

echo "ENV_FILE_PATH: ${ENV_FILE_PATH}"
echo "MODEL_TYPE: ${MODEL_TYPE}"
echo "WORKSPACE_PATH: ${WORKSPACE_PATH}"
echo "DATASET_PATHS: ${@:4}"

DATASET_ARGUMENT="--dataset-paths"
for DATASET_PATH in ${@:4}; do
    DATASET_ARGUMENT+=" ${DATASET_PATH}"
done

source ${ENV_FILE_PATH}

echo ">>> training a model for ${VOCAB}; model will be stored at ${WORKSPACE_PATH}"
time python -m training.run.train --model ${MODEL_TYPE} --workspace "${WORKSPACE_PATH}" ${DATASET_ARGUMENT}
