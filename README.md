# Video.catalyst
Example of using catalyst for video segmentation (classification) task using TSN network.

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
    --in-dir=/mnt/ssd1/datasets/video_dataset \
    --out-dataset=./data/dataset.csv \
    --out-labeling=./data/labeling.json \
    --tag-column=video
```

```bash
python tsn/process_data.py \
    --in-csv=./data/dataset.csv \
    --out-csv=./data/dataset_processed.csv \
    --n-cpu=16 --verbose \
    --out-dir=/mnt/ssd1/datasets/video_dataset_processed \
    --datapath=/mnt/ssd1/datasets/video_dataset
```

### Data splitting
```bash
PYTHONPATH=tsn python tsn/prepare_splits.py \
    --base_path=/mnt/ssd1/datasets/video_dataset \
    --dataset_path=./data/dataset.csv \
    --out_folder=data \
    --k_fold=8 \
    --use_folds=4
```
### Model training
```bash
CUDA_VISIBLE_DEVICES=1 catalyst-dl run --config configs/train.yml
```

# WIP
 
#### Tensorboard
```bash
CUDA_VISIBLE_DEVICES="" tensorboard --logdir tsn --ip=0.0.0.0 --port=6006
```

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

### Docker 
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
