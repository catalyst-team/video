[![Build Status](https://travis-ci.com/catalyst-team/classification.svg?branch=master)](https://travis-ci.com/catalyst-team/classification)
[![Telegram](https://img.shields.io/badge/news-on%20telegram-blue)](https://t.me/catalyst_team)
[![Gitter](https://badges.gitter.im/catalyst-team/community.svg)](https://gitter.im/catalyst-team/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Spectrum](https://img.shields.io/badge/chat-on%20spectrum-blueviolet)](https://spectrum.chat/catalyst)
[![Slack](https://img.shields.io/badge/ODS-slack-red)](https://opendatascience.slack.com/messages/CGK4KQBHD)
[![Donate](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/third_party_pics/patreon.png)](https://www.patreon.com/catalyst_team)

# Catalyst.Video

You will learn how to build video classification pipeline with transfer learning using the Catalyst framework to get reproducible results.

## Preparation
You have to split your video files into classes in the following format:
```bash
video_dataset/
    action_0/
        video_abc.avi
        ...
    action_1/
        video_abcd.avi
        ...
```
#### Data preprocessing

```bash
catalyst-data tag2label \
    --in-dir=./data/video_dataset \
    --out-dataset=./data/dataset.csv \
    --out-labeling=./data/labeling.json \
    --tag-column=video
```

```bash
python tsn/process_data.py \
    --in-csv=./data/dataset.csv \
    --datapath=./data/video_dataset \
    --out-csv=./data/dataset_processed.csv \
    --out-dir=./data/video_dataset_processed \
    --n-cpu=4 \
    --verbose
```

### Data splitting
```bash
PYTHONPATH=tsn python tsn/prepare_splits.py \
    --base_path=./data/video_dataset \
    --dataset_path=./data/dataset.csv \
    --out_folder=./data \
    --k_fold=5 \
    --use_folds=4
```
### Model training
```bash
CUDA_VISIBLE_DEVICES=0 catalyst-dl run --config configs/train.yml
```

 
#### Tensorboard
```bash
CUDA_VISIBLE_DEVICES="" tensorboard --logdir /mnt/ssd1/logs/tsn --host=0.0.0.0 --port=6006
```
# WIP
### Model inference
Don't forget to fix data_params at `infer.yaml`

```bash
export LOGDIR=/mnt/ssd1/logs/tsn
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=tsn catalyst-dl infer \
   --expdir=tsn \
   --resume=${LOGDIR}/checkpoint.best.pth.tar \
   --out-prefix=${LOGDIR}/dataset.predictions.npy \
   --config=${LOGDIR}/config.json,./configs/infer.yml \
   --verbose
 ```

### Docker  (TODO)
Build
```bash
docker build -t tsn .
```

Run
```bash
docker run --rm -it \
    -p 8893:8893 \
    --ipc=host \
    --runtime=nvidia \
    tsn \
    flask run -p 8893 --host=0.0.0.0
```
