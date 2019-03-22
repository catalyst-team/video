import argparse
from pathlib import Path
import os
import pandas as pd
import cv2
from sklearn.model_selection import GroupKFold, StratifiedKFold
from fold import StratifiedGroupKFold, get_labels_count
import numpy as np


def create_dataframe(base_path, dataset_path):
    df = pd.read_csv(dataset_path)
    df['folder'] = df.filepath.apply(lambda x: x.split('/')[-1].split('n')[0])
    df['frame_count'] = df.filepath.apply(lambda x: get_frame_count(base_path, x))
    df['frame_fps'] = df.filepath.apply(lambda x: get_frame_fps(base_path, x))
    df['secs'] = df.frame_count / df.frame_fps
    df['video_num'] = df.filepath.apply(get_video_num)
    return df


def get_video_num(p):
    return p.split('/')[-1].split('n')[0]


def get_frame_count(base_path, p):
    capture = cv2.VideoCapture(str(base_path / p))
    return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))


def get_frame_fps(base_path, p):
    capture = cv2.VideoCapture(str(base_path / p))
    return int(capture.get(cv2.CAP_PROP_FPS))


def get_folds(df, k, strategy):
    # for normal cross validation with using of all folds
    if strategy == 'GroupKFold':
        kfold = GroupKFold(k)
        splits = kfold.split(df.filepath, df.video, groups=df.video_num)
    elif strategy == 'StratifiedGroupKFold':
        kfold = StratifiedGroupKFold(k, n_jobs=10, max_combs=1e3, max_partitions=3)
        splits = kfold.split(df.filepath, df.video, groups=df.video_num, features=df.secs)
    elif strategy == 'StratifiedKFold':
        kfold = StratifiedKFold(k, random_state=42)
        splits = kfold.split(df.filepath, pd.factorize(df.frame_fps)[0])  # for example
    else:
        raise ValueError(f'Wrong value of strategy parameter: {strategy}')

    return splits


def get_split(df, k, strategy='GroupKFold', use_folds=1):
    # balancing train-val-test folds
    splits = get_folds(df, k, strategy)
    splits = [s[1] for s in splits]
    counts = []

    for split in splits:
        fold = df.iloc[split]
        counts.append(get_labels_count(fold.video, df.video.unique(), fold.secs))
    counts = np.array(counts)
    counts_mean = counts.mean(axis=0)
    counts_diff = (np.sqrt((counts - counts_mean) ** 2)).mean(axis=1)
    counts_diff_inds = np.argsort(counts_diff)

    splitted_folds = []
    for fold in range(use_folds):
        train_inds = []
        for i in counts_diff_inds:
            split = splits[i]
            if i == fold:  # choose one fold for validation
                val_inds = split
            else:
                train_inds.append(split)
        train_inds = np.hstack(train_inds)

        splitted_folds.append((df.iloc[train_inds], df.iloc[val_inds]))
    return splitted_folds


def path_to_videopath(path):
    return os.path.join(*str(path).split('/')[-2:]).split('.')[0]


def to_catalyst_format(df):
    df['path'] = df.path.apply(path_to_videopath)
    df = df.rename(columns={'label': 'video', 'path': 'videopath'})
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--k_fold', default=5, type=int)
    parser.add_argument('--out_folder', required=True)
    parser.add_argument('--use_folds', default=1, type=int)
    args = parser.parse_args()

    base_path = Path(args.base_path)
    out_folder = Path(args.out_folder)

    df = create_dataframe(base_path, args.dataset_path)
    df.filepath = df.filepath.apply(lambda x: x.split('.')[0])
    splitted_folds = get_split(df, args.k_fold, use_folds=args.use_folds)

    for i, (train_df, val_df) in enumerate(splitted_folds):
        train_df.to_csv(out_folder / f'train_fold_{i}.csv', index=False)
        val_df.to_csv(out_folder / f'val_fold_{i}.csv', index=False)


if __name__ == '__main__':
    main()
