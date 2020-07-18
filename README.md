# ww4ff
Wake word detection modeling for Firefox.

## Getting Started

* Clone the repository: `git clone https://github.com/castorini/ww4ff && cd ww4ff`

* [Install docker](https://docs.docker.com/engine/install/) and [enable GPU support](https://cnvrg.io/how-to-setup-docker-and-nvidia-docker-2-0-on-ubuntu-18-04/)

* `docker build -t ww4ff .`

* `./start.sh` will start up the docker. (if docker container is running it will use the same container) Note that /data and current directory will be mounted

* `conda activate ww4ff` 

* set `DATASET_PATH` env variable (`COMMON_VOICE_DATASET_PATH` and `WAKE_WORD_DATASET_PATH` might need to be set if processed dataset is missing)


## Pretraining the model
```
source envs/pretrain.env
NUM_LABELS=10 python -m ww4ff.run.pretrain --model res8 --workspace workspace/<workspace_name>
```

## Training res8 for hey firefox
```
source envs/res8.env
INFERENCE_SEQUENCE=[0,1] INFERENCE_WEIGHTS=[1,1,1,1,1,1,1,1,1,1] WW4FF_LOG_LEVEL=INFO NUM_LABELS=10 python -m ww4ff.run.train --model res8 --workspace workspace/<workspace_name> --load-weights --vocab " hey" fire fox

```

## Exporting the trained model as js file
```
python -m ww4ff.run.export_honkling -i workspace/<workspace_name>/model.pt -o <output_filename>.js --name RES8
```
