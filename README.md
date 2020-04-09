<div align="center">

[![Catalyst logo](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png)](https://github.com/catalyst-team/catalyst)

**Accelerated DL R&D**

[![Build Status](http://66.248.205.49:8111/app/rest/builds/buildType:id:Catalyst_Deploy/statusIcon.svg)](http://66.248.205.49:8111/project.html?projectId=Catalyst&tab=projectOverview&guest=1)
[![CodeFactor](https://www.codefactor.io/repository/github/catalyst-team/catalyst/badge)](https://www.codefactor.io/repository/github/catalyst-team/catalyst)
[![Pipi version](https://img.shields.io/pypi/v/catalyst.svg)](https://pypi.org/project/catalyst/)
[![Docs](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fcatalyst%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://catalyst-team.github.io/catalyst/index.html)
[![PyPI Status](https://pepy.tech/badge/catalyst)](https://pepy.tech/project/catalyst)

[![Twitter](https://img.shields.io/badge/news-on%20twitter-499feb)](https://twitter.com/catalyst_core)
[![Telegram](https://img.shields.io/badge/channel-on%20telegram-blue)](https://t.me/catalyst_team)
[![Slack](https://img.shields.io/badge/Catalyst-slack-success)](https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw)
[![Github contributors](https://img.shields.io/github/contributors/catalyst-team/catalyst.svg?logo=github&logoColor=white)](https://github.com/catalyst-team/catalyst/graphs/contributors)

</div>

PyTorch framework for Deep Learning research and development.
It was developed with a focus on reproducibility,
fast experimentation and code/ideas reusing.
Being able to research/develop something new,
rather than write another regular train loop. <br/>
Break the cycle - use the Catalyst!

Project [manifest](https://github.com/catalyst-team/catalyst/blob/master/MANIFEST.md). Part of [PyTorch Ecosystem](https://pytorch.org/ecosystem/). Part of [Catalyst Ecosystem](https://docs.google.com/presentation/d/1D-yhVOg6OXzjo9K_-IS5vSHLPIUxp1PEkFGnpRcNCNU/edit?usp=sharing):
- [Alchemy](https://github.com/catalyst-team/alchemy) - Experiments logging & visualization
- [Catalyst](https://github.com/catalyst-team/catalyst) - Accelerated Deep Learning Research and Development
- [Reaction](https://github.com/catalyst-team/reaction) - Convenient Deep Learning models serving

[Catalyst at AI Landscape](https://landscape.lfai.foundation/selected=catalyst).

----

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
