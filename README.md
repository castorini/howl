# Howl

[![PyPI](https://img.shields.io/pypi/v/howl?color=brightgreen)](https://pypi.org/project/howl/)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

Wake word detection modeling for Firefox Voice, supporting open datasets like Google Speech Commands and Mozilla Common Voice.

Citation:

```
@inproceedings{tang-etal-2020-howl,
    title = "Howl: A Deployed, Open-Source Wake Word Detection System",
    author = "Tang, Raphael and Lee, Jaejun and Razi, Afsaneh and Cambre, Julia and Bicking, Ian and Kaye, Jofish and Lin, Jimmy",
    booktitle = "Proceedings of Second Workshop for NLP Open Source Software (NLP-OSS)",
    month = nov,
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.nlposs-1.9",
    doi = "10.18653/v1/2020.nlposs-1.9",
    pages = "61--65"
}
```

## Quickstart Guide

1. Install PyAudio and [PyTorch 1.5+](https://pytorch.org) through your distribution's package system.

2. Install Howl using `pip`

```
pip install howl
```

3. To immediately use a pre-trained Howl model for inference, we provide the `client` API. The following example (also found under `examples/hey_fire_fox.py`) loads the "hey_fire_fox" pretrained model with a simple callback and starts the inference client.

```
from howl.client import HowlClient

def hello_callback(detected_words):
    print("Detected: {}".format(detected_words))

client = HowlClient()
client.from_pretrained("hey_fire_fox", force_reload=False)
client.add_listener(hello_callback)
client.start().join()
```

## Training Guide

### Installation

1. `git clone https://github.com/castorini/howl && cd howl`

2. Install [PyTorch](https://pytorch.org) by following your platform-specific instructions.

3. Install PyAudio and its dependencies through your distribution's package system.

4. `pip install -r requirements.txt -r requirements_training.txt` (some apt packages might need to be installed)

5. `./download_mfa.sh` to setup montreal forced alginer (MFA) for dataset generation

### Preparing a Dataset

Assuming MFA is installed using `download_mfa.sh` and [Common Voice dataset](https://commonvoice.mozilla.org/) is downloaded already, one can easily generate a dataset for custom wakeword using `generate_dataset.sh` script.
```bash
./generate_dataset.sh <common voice dataset path> <underscore separated wakeword (e.g. hey_fire_fox)> <inference sequence (e.g. [0,1,2])> <(Optional) "true" to skip negative dataset generation>
```

In the example that follows, we describe the process of generating a dataste for the word, "fire."

1. Download a supported data source. We recommend [Common Voice](https://commonvoice.mozilla.org/) for its breadth and free license.

2. To provide alignment for the data, install [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/stable/installation.html) (MFA)
and download an [English pronunciation dictionary](http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b).

3. Create a positive dataset containing the keyword:
```bash
VOCAB='["fire"]' INFERENCE_SEQUENCE=[0] DATASET_PATH=data/fire-positive python -m training.run.create_raw_dataset -i ~/path/to/common-voice --positive-pct 100 --negative-pct 0
```

4. Create a negative dataset without the keyword:
note that 5% is sufficient when generating negative dataset from common-voice dataset
```bash
VOCAB='["fire"]' INFERENCE_SEQUENCE=[0] DATASET_PATH=data/fire-negative python -m training.run.create_raw_dataset -i ~/path/to/common-voice --positive-pct 0 --negative-pct 5 
```

5. Generate some mock alignment for the negative set, where we don't care about alignment:

```bash
DATASET_PATH=data/fire-negative python -m training.run.attach_alignment --align-type stub
```

6. Use MFA to generate alignment for the positive set:

```bash
mfa_align data/fire-positive/audio eng.dict pretrained_models/english.zip output-folder
```

7. Attach the MFA alignment to the positive dataset:

```bash
DATASET_PATH=data/fire-positive python -m training.run.attach_alignment --align-type mfa -i output-folder
```

8. (Optional) Stitch vocab samples of aligned dataset to generate wakeword samples

```bash
VOCAB='["fire"]' INFERENCE_SEQUENCE=[0] python -m training.run.stitch_vocab_samples --aligned-dataset "data/fire-positive" --stitched-dataset "data/fire-stitched"
```

### Training and Running a Model

1. Source the relevant environment variables for training the `res8` model: `source envs/res8.env`.
2. Train the model: `python -m training.run.train -i data/fire-positive data/fire-negative data/fire-stitched --model res8 --workspace workspaces/fire-res8`.
3. For the CLI demo, run `python -m training.run.demo --model res8 --workspace workspaces/fire-res8`.

`train_model.sh` is also available which encaspulates individual command into a single bash script

```bash
./train_model.sh <env file path (e.g. envs/res8.env)> <model type (e.g. res8)> <workspace path (e.g. workspaces/fire-res8)> <dataset1 (e.g. data/fire-positive)> <dataset2(e.g. data/fire-negative)> ...
```

### Pretrained Models

[howl-models](https://github.com/castorini/howl-models) contains workspaces with pretrained models

To get the latest models, simply run `git submodule update --init --recursive`

- [hey firefox](https://github.com/castorini/howl-models/tree/master/howl/hey-fire-fox)

```bash
VOCAB='["hey","fire","fox"]' INFERENCE_SEQUENCE=[0,1,2] INFERENCE_THRESHOLD=0 NUM_MELS=40 MAX_WINDOW_SIZE_SECONDS=0.5 python -m training.run.demo --model res8 --workspace howl-models/howl/hey-fire-fox
```

## Reproducing Paper Results

First, follow the installation instructions in the quickstart guide.

### Google Speech Commands

1. Download [the Google Speech Commands dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) and extract it.
2. Source the appropriate environment variables: `source envs/res8.env`
3. Set the dataset path to the root folder of the Speech Commands dataset: `export DATASET_PATH=/path/to/dataset`
4. Train the `res8` model: `NUM_EPOCHS=20 MAX_WINDOW_SIZE_SECONDS=1 VOCAB='["yes","no","up","down","left","right","on","off","stop","go"]' BATCH_SIZE=64 LR_DECAY=0.8 LEARNING_RATE=0.01 python -m training.run.pretrain_gsc --model res8`

### Hey Firefox

1. Download [the Hey Firefox corpus](https://nlp.nyc3.digitaloceanspaces.com/hey-ff-data.zip), licensed under CC0, and extract it.
2. Download [our noise dataset](https://nlp.nyc3.digitaloceanspaces.com/hey-ff-noise.zip), built from Microsoft SNSD and MUSAN, and extract it.
3. Source the appropriate environment variables: `source envs/res8.env`
4. Set the noise dataset path to the root folder: `export NOISE_DATASET_PATH=/path/to/snsd`
5. Set the firefox dataset path to the root folder: `export DATASET_PATH=/path/to/hey_firefox`
6. Train the model: `LR_DECAY=0.98 VOCAB='["hey","fire","fox"]' USE_NOISE_DATASET=True BATCH_SIZE=16 INFERENCE_THRESHOLD=0 NUM_EPOCHS=300 NUM_MELS=40 INFERENCE_SEQUENCE=[0,1,2] MAX_WINDOW_SIZE_SECONDS=0.5 python -m training.run.train --model res8 --workspace workspaces/hey-ff-res8`

### Hey Snips

1. Download [hey snips dataset](https://github.com/sonos/keyword-spotting-research-datasets)
2. Process the dataset to a format howl can load

```bash
VOCAB='["hey","snips"]' INFERENCE_SEQUENCE=[0,1] DATASET_PATH=data/hey-snips python -m training.run.create_raw_dataset --dataset-type 'hey-snips' -i ~/path/to/hey_snips_dataset
```

3. Generate some mock alignment for the dataset, where we don't care about alignment:

```bash
DATASET_PATH=data/hey-snips python -m training.run.attach_alignment --align-type stub
```

4. Use MFA to generate alignment for the dataset set:

```bash
mfa_align data/hey-snips/audio eng.dict pretrained_models/english.zip output-folder
```

5. Attach the MFA alignment to the dataset:

```bash
DATASET_PATH=data/hey-snips python -m training.run.attach_alignment --align-type mfa -i output-folder
```

6. Source the appropriate environment variables: `source envs/res8.env`
7. Set the noise dataset path to the root folder: `export NOISE_DATASET_PATH=/path/to/snsd`
8. Set the noise dataset path to the root folder: `export DATASET_PATH=/path/to/hey-snips`
9. Train the model: `LR_DECAY=0.98 VOCAB='["hey","snips"]' USE_NOISE_DATASET=True BATCH_SIZE=16 INFERENCE_THRESHOLD=0 NUM_EPOCHS=300 NUM_MELS=40 INFERENCE_SEQUENCE=[0,1] MAX_WINDOW_SIZE_SECONDS=0.5 python -m training.run.train --model res8 --workspace workspaces/hey-snips-res8`

### Generating dataset for Mycroft-precise

howl also provides a script for transforming howl dataset to [mycroft-precise](https://github.com/MycroftAI/mycroft-precise) dataset
```bash
VOCAB='["hey","fire","fox"]' INFERENCE_SEQUENCE=[0,1,2] python -m training.run.generate_precise_dataset --dataset-path /path/to/howl_dataset
```

## Experiments

To verify the correctness of our implementation, we first train and evaluate our models on the Google Speech Commands dataset, for which there exists many known results. Next, we curate a wake word detection datasets and report our resulting model quality.

For both experiments, we generate reports in excel format. [experiments](https://github.com/castorini/howl/tree/master/experiments) folder includes sample outputs from the for each experiment and corresponding workspaces can be found [here](https://github.com/castorini/howl-models/tree/master/howl/experiments)

### commands_recognition

For command recognition, we train the four different models (res8, LSTM, LAS encoder, MobileNetv2) to detect twelve different keywords: “yes”, “no”, “up”, “down”, “left”, “right”, “on”, “off”, “stop”, “go”, unknown, or silence.

```bash
python -m training.run.eval_commands_recognition --num_iterations n --dataset_path < path_to_gsc_datasets >
```

### word_detection

In this experiment, we train our best commands recognition model, res8, for `hey firefox` and `hey snips` and evaluate them with different threashold.

Two different performance reports are generated, one with the clean audio and one with audios with noise

```bash
python -m training.run.eval_wake_word_detection --num_models n --hop_size < number between 0 and 1 > --exp_type < hey_firefox | hey_snips > --dataset_path "x" --noiseset_path "y"
```

We also provide a script for generating ROC curve. `exp_timestamp` can be found from the reports generated from previous command

```bash
python -m training.run.generate_roc --exp_timestamp < experiment timestamp > --exp_type < hey_firefox | hey_snips >
```
