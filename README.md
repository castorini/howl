# Howl
Wake word detection modeling for Firefox Voice, supporting open datasets like Google Speech Commands and Mozilla Common Voice.

Citation:
```
@article{tang2020howl,
  title={Howl: A Deployed, Open-Source Wake Word Detection System},
  author={Raphael Tang and Jaejun Lee and Afsaneh Razi and Julia Cambre and Ian Bicking and Jofish Kaye and Jimmy Lin},
  journal={arXiv:2008.09606},
  year={2020}
}
```

## Quickstart Guide

### Installation

A proper Pip package is coming soon. 

1. `git clone https://github.com/castorini/howl && cd howl`

2. Install [PyTorch](https://pytorch.org) by following your platform-specific instructions.

3. Install PyAudio and its dependencies through your distribution's package system.

4. `pip install -r requirements.txt`

### Preparing a Dataset

In the example that follows, we describe how to train a custom detector for the word, "fire."

1. Download a supported data source. We recommend [Common Voice](https://commonvoice.mozilla.org/) for its breadth and free license.
2. To provide alignment for the data, install [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html) (MFA)
and download an [English pronunciation dictionary](http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b).
3. Create a positive dataset containing the keyword: 
```bash
VOCAB='["fire"]' INFERENCE_SEQUENCE=[0] DATASET_PATH=data/fire-positive python -m howl.run.create_raw_dataset --negative-pct 0 -i ~/path/to/common-voice --positive-pct 100
```
4. Create a negative dataset without the keyword:
```bash
VOCAB='["fire"]' INFERENCE_SEQUENCE=[0] DATASET_PATH=data/fire-negative python -m howl.run.create_raw_dataset --negative-pct 5 -i ~/path/to/common-voice --positive-pct 0
```
5. Generate some mock alignment for the negative set, where we don't care about alignment:
```bash
DATASET_PATH=data/fire-negative python -m howl.run.attach_alignment --align-type stub
```
6. Use MFA to generate alignment for the positive set:
```bash
mfa_align data/fire-positive/audio eng.dict pretrained_models/english.zip output-folder
```
7. Attach the MFA alignment to the positive dataset:
```bash
DATASET_PATH=data/fire-positive python -m howl.run.attach_alignment --align-type mfa -i output-folder
```

### Training and Running a Model

1. Source the relevant environment variables for training the `res8` model: `source envs/res8.env`.
2. Train the model: `python -m howl.run.train -i data/fire-negative data/fire-positive --model res8 --workspace workspaces/fire-res8`.
3. For the CLI demo, run `python -m howl.run.demo --model res8 --workspace workspaces/fire-res8`.

## Reproducing Paper Results

First, follow the installation instructions in the quickstart guide.

### Google Speech Commands

1. Download [the Google Speech Commands dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) and extract it.
2. Source the appropriate environment variables: `source envs/res8.env`
3. Set the dataset path to the root folder of the Speech Commands dataset: `export DATASET_PATH=/path/to/dataset`
4. Train the `res8` model: `NUM_EPOCHS=20 MAX_WINDOW_SIZE_SECONDS=1 VOCAB='["yes","no","up","down","left","right","on","off","stop","go"]' BATCH_SIZE=64 LR_DECAY=0.8 LEARNING_RATE=0.01 python -m howl.run.pretrain_gsc --model res8`

### Hey Firefox

1. Download [the Hey Firefox corpus](http://nlp.rocks/firefox), licensed under CC0, and extract it.
2. Download [our noise dataset](http://nlp.rocks/ffnoise), built from Microsoft SNSD and MUSAN, and extract it.
3. Source the appropriate environment variables: `source envs/res8.env`
4. Set the noise dataset path to the root folder: `export NOISE_DATASET_PATH=/path/to/snsd`
5. Train the model: `LR_DECAY=0.98 VOCAB='[" hey","fire","fox"]' USE_NOISE_DATASET=True BATCH_SIZE=16 INFERENCE_THRESHOLD=0 NUM_EPOCHS=300 NUM_MELS=40 INFERENCE_SEQUENCE=[0,1,2] MAX_WINDOW_SIZE_SECONDS=0.5 python -m howl.run.train --model res8 --workspace workspaces/hey-ff-res8 -i /path/to/hey_firefox`

### Hey Snips

1. Download [hey snips dataset](https://github.com/sonos/keyword-spotting-research-datasets)
2. Process the dataset to a format howl can load
```bash
VOCAB='["hey" "snips"]' INFERENCE_SEQUENCE=[0,1] DATASET_PATH=data/hey-snips python -m howl.run.create_raw_dataset -i ~/path/to/hey_snips_dataset
```
3. Use MFA to generate alignment for the dataset set:
```bash
mfa_align data/hey-snips/audio eng.dict pretrained_models/english.zip output-folder
```
4. Attach the MFA alignment to the dataset:
```bash
DATASET_PATH=data/hey-snips python -m howl.run.attach_alignment --align-type mfa -i output-folder
```
5. Source the appropriate environment variables: `source envs/res8.env`
6. Set the noise dataset path to the root folder: `export NOISE_DATASET_PATH=/path/to/snsd`
7. Train the model: `LR_DECAY=0.98 VOCAB='[" hey","snips"]' USE_NOISE_DATASET=True BATCH_SIZE=16 INFERENCE_THRESHOLD=0 NUM_EPOCHS=300 NUM_MELS=40 INFERENCE_SEQUENCE=[0,1] MAX_WINDOW_SIZE_SECONDS=0.5 python -m howl.run.train --model res8 --workspace workspaces/hey-snips-res8 -i /path/to/hey-snips`