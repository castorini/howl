#bin/bash
set -e
set -v

COMMON_VOICE_DATASET_PATH=${1} # common voice dataset path
DATASET_NAME=${2} # underscore separated wakeword (e.g. hey_fire_fox)
INFERENCE_SEQUENCE=${3} # inference sequence (e.g. [0,1,2])
# ${4} pass true to skip generating negative dataset

if [ $# -lt 3 ]; then
    printf 1>&2 "invalid arguments: ./generate_dataset.sh <common voice dataset path> <underscore separated wakeword> <inference sequence>"
    exit 2
elif [ $# -eq 4 ]; then
    SKIP_NEG_DATASET=${4}
else
    SKIP_NEG_DATASET="false"
fi

printf "COMMON_VOICE_DATASET_PATH: ${COMMON_VOICE_DATASET_PATH}\n"
printf "DATASET_NAME: ${DATASET_NAME}\n"
printf "INFERENCE_SEQUENCE: ${INFERENCE_SEQUENCE}\n"

VOCAB="["
IFS='_'
read -ra ADDR <<< "${DATASET_NAME}"
for i in "${ADDR[@]}"; do
    VOCAB+="\"${i}\","
done
VOCAB="${VOCAB::-1}]"

DATASET_FOLDER="datasets"
printf "\n\n>>> generating datasets for ${VOCAB} at ${DATASET_FOLDER}\n"
mkdir -p "${DATASET_FOLDER}"

NEGATIVE_PCT=0
if [ ${SKIP_NEG_DATASET} != "true" ]; then
    printf "\n\n>>> dataset with negative samples will also be generated\n"
    NEGATIVE_PCT=5
fi

printf "\n\n>>> generating raw audio dataset\n"
mkdir -p "${DATASET_FOLDER}"
time VOCAB=${VOCAB} INFERENCE_SEQUENCE=${INFERENCE_SEQUENCE} python -m training.run.generate_raw_audio_dataset -i ${COMMON_VOICE_DATASET_PATH} --positive-pct 100 --negative-pct ${NEGATIVE_PCT} --overwrite true

NEG_DATASET_PATH="${DATASET_FOLDER}/${DATASET_NAME}/negative"
POS_DATASET_PATH="${DATASET_FOLDER}/${DATASET_NAME}/positive"

POS_DATASET_ALIGNMENT="${POS_DATASET_PATH}/alignment"
printf "\n\n>>> generating alignment for the positive dataset using MFA: ${POS_DATASET_ALIGNMENT}\n"
mkdir -p "${POS_DATASET_ALIGNMENT}"
MFA_FOLDER="./montreal-forced-aligner"
pushd ${MFA_FOLDER}
# yes n for "There were words not found in the dictionary. Would you like to abort to fix them? (Y/N)"
# if process fails, check ~/Documents/MFA/audio/train/mfcc/log/make_mfcc.0.log
time yes n | ./bin/mfa_align --verbose --clean --num_jobs 12 "../${POS_DATASET_PATH}/audio" librispeech-lexicon.txt pretrained_models/english.zip "../${POS_DATASET_ALIGNMENT}"
popd

printf "\n\n>>> attaching the MFA alignment to the positive dataset\n"
time python -m training.run.attach_alignment \
  --input-raw-audio-dataset "${POS_DATASET_PATH}" \
  --token-type word \
  --alignment-type mfa \
  --alignments-path "${POS_DATASET_ALIGNMENT}"

if [ ${SKIP_NEG_DATASET} != "true" ]; then
    printf "\n\n>>> attaching mock alignment to the negative dataset\n"
    time python -m training.run.attach_alignment \
      --alignment-type stub \
      --input-raw-audio-dataset "${NEG_DATASET_PATH}" \
      --token-type word
fi

printf "\n\n>>> stitching vocab samples to generate a dataset made up of stitched wakeword samples\n"
time VOCAB=${VOCAB} INFERENCE_SEQUENCE=${INFERENCE_SEQUENCE} python -m training.run.stitch_vocab_samples --dataset-path "${POS_DATASET_PATH}"

printf "\n\n>>> Dataset is ready for ${VOCAB}\n"
