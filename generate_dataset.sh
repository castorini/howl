#bin/bash
set -e

DATASET_NAME=${1} # _ separated wakeword (hey_fire_fox)
COMMON_VOICE_DATASET_PATH=${2} # common voice dataset path

if [ $# -lt 2 ]; then
  echo 1>&2 "invalid arguments: ./generate_dataset.sh <_ separated wakeword> <common voice dataset path>"
  exit 2
fi

VOCAB="["
IFS='_'
read -ra ADDR <<< "${DATASET_NAME}"
for i in "${ADDR[@]}"; do 
    VOCAB+="\"${i}\","
done
VOCAB="${VOCAB::-1}]"

DATASET_FOLDER="data/${DATASET_NAME}"
echo ">>> generating datasets for ${VOCAB} at ${DATASET_FOLDER}"

mkdir -p ${DATASET_FOLDER}
export COMMON_VOICE_DATASET_PATH="/data/common-voice/"

NEG_DATASET_PATH="${DATASET_FOLDER}/negative"
echo ">>> generating negative dataset: ${NEG_DATASET_PATH}"
time DATASET_PATH=${NEG_DATASET_PATH} python -m training.run.create_raw_dataset --negative-pct 5 -i ${COMMON_VOICE_DATASET_PATH} --positive-pct 0

echo ">>> generating mock alignment for the negative set"
time DATASET_PATH=${NEG_DATASET_PATH} python -m training.run.attach_alignment --align-type stub

POS_DATASET_PATH="${DATASET_FOLDER}/positive"
echo ">>> generating positive dataset: ${POS_DATASET_PATH}"
time DATASET_PATH=${POS_DATASET_PATH} python -m training.run.create_raw_dataset --negative-pct 0 -i ${COMMON_VOICE_DATASET_PATH} --positive-pct 100

POS_DATASET_ALIGNMENT="${POS_DATASET_PATH}/alignment"
echo ">>> generating alignment for the positive dataset using MFA: ${POS_DATASET_ALIGNMENT}"
MFA_FOLDER="./montreal-forced-aligner"
pushd ${MFA_FOLDER}
# yes n for "There were words not found in the dictionary. Would you like to abort to fix them? (Y/N)"
# if process fails, check ~/Documents/MFA/audio/train/mfcc/log/make_mfcc.0.log
# it's often due to missing openblas or fortran packages
# if this is the case, simply install them using apt
time yes n | ./bin/mfa_align --verbose --clean --num_jobs 12 ../${POS_DATASET_PATH}/audio librispeech-lexicon.txt pretrained_models/english.zip ../${POS_DATASET_ALIGNMENT}
popd

echo ">>> attaching the MFA alignment to the positive dataset"
DATASET_PATH=${POS_DATASET_PATH} python -m training.run.attach_alignment --align-type mfa -i ${POS_DATASET_ALIGNMENT}

echo ">>> Dataset is ready for ${VOCAB}"