# Howl
Wake word detection modeling for Firefox Voice, supporting open datasets like Google Speech Commands and Mozilla Common Voice.

## Quickstart Guide

### Installation

A proper Pip package is coming soon. 

1. `git clone https://github.com/castorini/howl && cd howl`

2. Install [PyTorch](https://pytorch.org) by following your platform-specific instructions.

3. `pip install -r requirements.txt`

### Preparing a Dataset

In the example that follows, we describe how to train a custom detector for the word, "fire."

1. Download a supported data source. We recommend [Common Voice](https://commonvoice.mozilla.org/) for its breadth and free license.
2. To provide alignment for the data, install [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html) (MFA)
and download an [English pronunciation dictionary](http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b).
3. Create a positive dataset containing the keyword: 
```bash
INFERENCE_SEQUENCE=[0] DATASET_PATH=data/fire-positive python -m howl.run.create_raw_dataset --negative-pct 0 --vocab fire -i ~/path/to/common-voice --positive-pct 100`
```
4. Create a negative dataset without the keyword:
```bash
INFERENCE_SEQUENCE=[0] DATASET_PATH=data/fire-negative python -m howl.run.create_raw_dataset --negative-pct 5 --vocab fire -i ~/path/to/common-voice --positive-pct 0`
```
5. Generate some mock alignment for the negative set, since we don't care about alignment for the negative set:
```bash
DATASET_PATH=data/fire-negative python -m howl.run.attach_alignment --align-type stub
```
6. Use MFA to generate alignment for the positive set:
```bash
mfa_align data/fire-positive eng.dict pretrained_models/english.zip output-folder
```
7. Attach the MFA alignment to the positive dataset:
```bash
DATASET_PATH=data/fire-positive python -m howl.run.attach_alignment --align-type mfa -i output-folder
```

### Training and Running a Model

1. Source the relevant environment variables for training the `res8` model: `source envs/res8.env`.
2. Train the model: `python -m howl.run.train -i data/fire-negative data/fire-positive --model res8 --vocab fire --workspace workspaces/fire-res8`.
3. For the CLI demo, run `python -m howl.run.demo --model res8 --workspace workspaces/fire-res8 --vocab fire`.