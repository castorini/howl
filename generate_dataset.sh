#bin/bash
set -e

COMMON_VOICE_DATASET_PATH=${1} # common voice dataset path
DATASET_NAME=${2} # underscore separated wakeword (e.g. hey_fire_fox)
INFERENCE_SEQUENCE=${3} # inference sequence (e.g. [0,1,2])
#${4} pass true to skip generating negative dataset

if [ $# -lt 3 ]; then
    echo 1>&2 "invalid arguments: ./generate_dataset.sh <common voice dataset path> <underscore separated wakeword> <inference sequence>"
    exit 2
elif [ $# -eq 4 ]; then
    SKIP_NEG_DATASET=${4}
else
    SKIP_NEG_DATASET="false"
fi

echo "COMMON_VOICE_DATASET_PATH: ${COMMON_VOICE_DATASET_PATH}"
echo "DATASET_NAME: ${DATASET_NAME}"
echo "INFERENCE_SEQUENCE: ${INFERENCE_SEQUENCE}"

VOCAB="["
IFS='_'
read -ra ADDR <<< "${DATASET_NAME}"
for i in "${ADDR[@]}"; do 
    VOCAB+="\"${i}\","
done
VOCAB="${VOCAB::-1}]"

DATASET_FOLDER="data/${DATASET_NAME}"
echo ">>> generating datasets for ${VOCAB} at ${DATASET_FOLDER}"
mkdir -p "${DATASET_FOLDER}"

if [ ${SKIP_NEG_DATASET} != "true" ]; then
    NEG_DATASET_PATH="${DATASET_FOLDER}/negative"
    echo ">>> generating negative dataset: ${NEG_DATASET_PATH}"
    mkdir -p "${NEG_DATASET_PATH}"
    time VOCAB=${VOCAB} INFERENCE_SEQUENCE=${INFERENCE_SEQUENCE} DATASET_PATH=${NEG_DATASET_PATH} python -m training.run.create_raw_dataset -i ${COMMON_VOICE_DATASET_PATH} --positive-pct 0 --negative-pct 5

    echo ">>> generating mock alignment for the negative set"
    time DATASET_PATH=${NEG_DATASET_PATH} python -m training.run.attach_alignment --align-type stub
fi

POS_DATASET_PATH="${DATASET_FOLDER}/positive"
echo ">>> generating positive dataset: ${POS_DATASET_PATH}"
mkdir -p "${POS_DATASET_PATH}"
time VOCAB=${VOCAB} INFERENCE_SEQUENCE=${INFERENCE_SEQUENCE} DATASET_PATH=${POS_DATASET_PATH} python -m training.run.create_raw_dataset -i ${COMMON_VOICE_DATASET_PATH} --positive-pct 100 --negative-pct 0

POS_DATASET_ALIGNMENT="${POS_DATASET_PATH}/alignment"
echo ">>> generating alignment for the positive dataset using MFA: ${POS_DATASET_ALIGNMENT}"
mkdir -p "${POS_DATASET_ALIGNMENT}"
MFA_FOLDER="./montreal-forced-aligner"
pushd ${MFA_FOLDER}
# yes n for "There were words not found in the dictionary. Would you like to abort to fix them? (Y/N)"
# if process fails, check ~/Documents/MFA/audio/train/mfcc/log/make_mfcc.0.log
# it's often due to missing openblas or fortran packages
# if this is the case, simply install them using apt
time yes n | ./bin/mfa_align --verbose --clean --num_jobs 12 "../${POS_DATASET_PATH}/audio" librispeech-lexicon.txt pretrained_models/english.zip "../${POS_DATASET_ALIGNMENT}"
popd

echo ">>> attaching the MFA alignment to the positive dataset"
time DATASET_PATH=${POS_DATASET_PATH} python -m training.run.attach_alignment --align-type mfa -i "${POS_DATASET_ALIGNMENT}"

STITCHED_DATASET="${DATASET_FOLDER}/stitched"
echo ">>> stitching vocab samples to generate a datset made up of stitched wakeword samples: ${STITCHED_DATASET}"
time VOCAB=${VOCAB} INFERENCE_SEQUENCE=${INFERENCE_SEQUENCE} python -m training.run.stitch_vocab_samples --aligned-dataset "${POS_DATASET_PATH}" --stitched-dataset "${STITCHED_DATASET}"

echo ">>> Dataset is ready for ${VOCAB}"
