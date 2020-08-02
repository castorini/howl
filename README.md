# howl
Wake word detection modeling using Mozilla Common Voice for Firefox Voice.

## Getting Started

* Clone the repository: `git clone https://github.com/castorini/howl && cd howl`

* [Install docker](https://docs.docker.com/engine/install/) and [enable GPU support](https://cnvrg.io/how-to-setup-docker-and-nvidia-docker-2-0-on-ubuntu-18-04/)

* `docker build -t howl .`

* `./start.sh` will start up the docker. (if docker container is running it will use the same container) Note that /data and current directory will be mounted

* `conda activate howl` 

* set `DATASET_PATH` env variable (`COMMON_VOICE_DATASET_PATH` and `WAKE_WORD_DATASET_PATH` might need to be set if processed dataset is missing)


## Pretraining the model
```
source envs/pretrain.env
NUM_LABELS=10 python -m howl.run.pretrain --model res8 --workspace workspace/<workspace_name>
```

## Training res8 for hey firefox
```
source envs/res8.env
INFERENCE_SEQUENCE=[0,1] INFERENCE_WEIGHTS=[1,1,1,1,1,1,1,1,1,1] WW4FF_LOG_LEVEL=INFO NUM_LABELS=10 python -m howl.run.train --model res8 --workspace workspace/<workspace_name> --load-weights --vocab " hey" "fire fox"

```

## Exporting the trained model as js file
```
python -m howl.run.export_honkling -i workspace/<workspace_name>/model-best.pt -o <output_filename>.js --name RES8
```
