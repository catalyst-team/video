import os
import numpy as np
import random
import cv2
import collections
import torch
from catalyst.dl.datasource import AbstractDataSource
from catalyst.data.sampler import BalanceClassSampler
from catalyst.data.reader import ScalarReader, ReaderCompose
from catalyst.data.augmentor import Augmentor
from catalyst.data.functional import read_image
from catalyst.dl.utils import UtilsFactory
from catalyst.legacy.utils.parse import parse_in_csvs

from albumentations import (
    Resize, JpegCompression, Normalize,
    HorizontalFlip, ShiftScaleRotate,
    HueSaturationValue, MotionBlur,
    RandomBrightnessContrast, BasicTransform,
    OpticalDistortion, CLAHE)


class ToTorchTensor(BasicTransform):
    def apply(self, img, **params):
        return torch.tensor(img).permute(2, 0, 1)


class GroupTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        for tr in self.transforms:
            p = getattr(tr, "p", 1.0)
            if random.random() < p:
                params = getattr(tr, "get_params", lambda: {})()
                images = [tr.apply(x, **params) for x in images]
        return images


class TorchStack:
    def __call__(self, tensors):
        tensors = torch.stack(tensors)
        return tensors


def preprocess_val_data(data):
    new_data = []
    for d in data:
        for start_interval in range(int(d['secs'])):
            d['offset'] = start_interval
            new_data.append(d)
    return new_data


class VideoImageReader:
    """
    Video images reader abstraction.
    """
    def __init__(
            self,
            input_key: str,
            output_key: str,
            datapath: str = None,
            grayscale: bool = False,
            n_frames=None,
            n_segments=None,
            time_window=None,
            uniform_time_sample=False,
            with_offset=False):
        """
        :param input_key: input key to use from annotation dict
        :param output_key: output key to use to store the result
        :param datapath: path to images dataset
            (so your can use relative paths in annotations)
        :param grayscale: boolean flag
            if you need to work only with grayscale images
        :param n_frames: number of frames to take from each segment
        :param n_segments: number of frames splitting
        :param time_window: length of sample crop from video  in seconds
        :param uniform_time_sample: use uniform frames sample from time interval or not
        """
        self.input_key = input_key
        self.output_key = output_key
        self.datapath = datapath
        self.grayscale = grayscale
        self.n_frames = n_frames
        self.n_segments = n_segments
        self.time_window = time_window
        self.uniform_time_sample = uniform_time_sample
        self.with_offset = with_offset

    def __call__(self, row):
        file_dir = str(row[self.input_key])
        if self.datapath is not None:
            file_dir = (
                file_dir
                if file_dir.startswith(self.datapath)
                else os.path.join(self.datapath, file_dir))

        frames = [os.path.join(file_dir, img) for img in os.listdir(file_dir)]
        fps = int(frames[0].split('/')[-1].split('_')[1].split('.')[0])

        # Step 1: sorting
        frames = sorted(frames)

        # Step 2: sampling by time
        if self.time_window:
            frame_window = self.time_window * fps
            if self.with_offset:
                start_frame = row['offset'] * fps
            elif len(frames) > frame_window:
                start_frame = np.random.randint(0, len(frames) - frame_window)
            else:
                start_frame = 0
            frames = frames[start_frame: start_frame + frame_window]  # mb less than frame_window
        else:
            frame_window = len(frames)

        # Step 3: choosing frames from time range
        # Option a: uniform time sampling (with constant time step)
        if self.uniform_time_sample:
            frames_indexes = [int(frame_window / self.n_frames * i) for i in range(self.n_frames)]
            # If frames less than needed - duplicate last. Important to keep constant time step
            frames = [frames[min(i, len(frames) - 1)] for i in frames_indexes]

        # Option b: (self.n_frames is None) use all frames (bad idea)
        elif self.n_frames is not None:
            tmp_frames = []
            if self.n_segments is not None:
                # Option c: random n sample from each successive interval
                frames = np.array_split(frames, self.n_segments)
            else:
                # Option d: random n sample (at all) from all frames
                frames = [frames]

            for frames_ in frames:
                replace_ = self.n_frames > len(frames_)
                frames_ = np.random.choice(
                    frames_,
                    self.n_frames,
                    replace=replace_).tolist()
                tmp_frames.extend(frames_)

            tmp_frames = sorted(tmp_frames)  # because of random.choice
            frames = tmp_frames

        frames = [
            read_image(x, datapath=self.datapath, grayscale=self.grayscale)
            for x in frames]
        result = {self.output_key: frames}
        return result


class DataSource(AbstractDataSource):

    @staticmethod
    def prepare_transforms(
            *,
            mode,
            stage=None,
            input_size=None,
            **kwargs):
        AbstractDataSource.prepare_transforms(mode=mode, stage=stage, **kwargs)

        input_size = input_size or 224

        # train_crop = GroupMultiScaleCrop(input_size)
        train_image_transforms = [
            OpticalDistortion(distort_limit=0.3, p=0.3),
            JpegCompression(quality_lower=50, p=0.8),
            HorizontalFlip(p=0.5),
            MotionBlur(p=0.5),
            ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),
            RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.4),
            HueSaturationValue(hue_shift_limit=3, sat_shift_limit=20, val_shift_limit=30, p=0.4),
            CLAHE(clip_limit=2, p=0.3)
        ]
        infer_image_transforms = [
            Resize(input_size, input_size),
            Normalize(),
            ToTorchTensor(p=1.0)
        ]
        stack = TorchStack()

        train_images_fn = GroupTransform(
            transforms=train_image_transforms + infer_image_transforms)
        valid_images_fn = GroupTransform(
            transforms=infer_image_transforms)

        def train_aug_fn(images):
            images = train_images_fn(images)
            images = stack(images)
            return images

        def valid_aug_fn(images):
            images = valid_images_fn(images)
            images = stack(images)
            return images

        train_transforms = Augmentor(
                dict_key="features",
                augment_fn=lambda x: train_aug_fn(x))

        valid_transforms = Augmentor(
            dict_key="features",
            augment_fn=lambda x: valid_aug_fn(x))

        if mode == "train":
            return train_transforms
        else:
            return valid_transforms

    @staticmethod
    def prepare_loaders(
            *,
            mode=None,
            stage=None,
            n_workers: int = None,
            batch_size: int = None,
            datapath=None,
            in_csv=None,
            in_csv_train=None,
            in_csv_valid=None,
            in_csv_infer=None,
            train_folds=None,
            valid_folds=None,
            tag2class=None,
            class_column=None,
            tag_column=None,
            one_hot_classes=None,
            folds_seed=42,
            n_folds=5,
            n_frames=None,
            n_segments=None,
            time_window=None,
            uniform_time_sample=False,
            train_frac=1,
    ):
        df, df_train, df_valid, df_infer = parse_in_csvs(
            in_csv=in_csv,
            in_csv_train=in_csv_train,
            in_csv_valid=in_csv_valid,
            in_csv_infer=in_csv_infer,
            train_folds=train_folds,
            valid_folds=valid_folds,
            tag2class=tag2class,
            class_column=class_column,
            tag_column=tag_column,
            folds_seed=folds_seed,
            n_folds=n_folds)

        df_valid = preprocess_val_data(df_valid)
        assert 0 < train_frac <= 1
        if train_frac < 1:
            train_size = int(len(df_train) * train_frac)
            df_train = df_train[:train_size]

        loaders = collections.OrderedDict()

        open_fn = [
            ScalarReader(
                input_key="class",
                output_key="targets",
                default_value=-1,
                dtype=np.int64),

        ]
        if one_hot_classes is not None:
            open_fn.append(
                ScalarReader(
                    input_key="class",
                    output_key="targets_onehot",
                    default_value=-1,
                    dtype=np.int32,
                    one_hot_classes=one_hot_classes))

        open_fn_val = open_fn.copy()
        open_fn.append(
            VideoImageReader(
                input_key="filepath",
                output_key="features",
                datapath=datapath,
                n_frames=n_frames,
                n_segments=n_segments,
                time_window=time_window,
                uniform_time_sample=uniform_time_sample))
        open_fn_val.append(
            VideoImageReader(
                input_key="filepath",
                output_key="features",
                datapath=datapath,
                n_frames=n_frames,
                n_segments=n_segments,
                time_window=time_window,
                uniform_time_sample=uniform_time_sample,
                with_offset=True))

        open_fn = ReaderCompose(readers=open_fn)
        open_fn_val = ReaderCompose(readers=open_fn_val)

        if len(df_train) > 0:
            labels = [x["class"] for x in df_train]
            sampler = BalanceClassSampler(labels, mode="upsampling")
            dict_transform = DataSource.prepare_transforms(
                mode="train", stage=stage)

            train_loader = UtilsFactory.create_loader(
                data_source=df_train,
                open_fn=open_fn,
                dict_transform=dict_transform,
                dataset_cache_prob=-1,
                batch_size=batch_size,
                workers=n_workers,
                shuffle=sampler is None,
                sampler=sampler)

            print("Train samples", len(train_loader) * batch_size)
            print("Train batches", len(train_loader))
            loaders["train"] = train_loader

        if len(df_valid) > 0:
            dict_transform = DataSource.prepare_transforms(
                mode="valid", stage=stage)
            valid_loader = UtilsFactory.create_loader(
                data_source=df_valid,
                open_fn=open_fn_val,
                dict_transform=dict_transform,
                dataset_cache_prob=-1,
                batch_size=batch_size,
                workers=n_workers,
                shuffle=False,
                sampler=None)

            print("Valid samples", len(valid_loader) * batch_size)
            print("Valid batches", len(valid_loader))
            loaders["valid"] = valid_loader

        if len(df_infer) > 0:
            dict_transform = DataSource.prepare_transforms(
                mode="infer", stage=stage)
            infer_loader = UtilsFactory.create_loader(
                data_source=df_infer,
                open_fn=open_fn,
                dict_transform=dict_transform,
                dataset_cache_prob=-1,
                batch_size=batch_size,
                workers=n_workers,
                shuffle=False,
                sampler=None)

            print("Infer samples", len(infer_loader) * batch_size)
            print("Infer batches", len(infer_loader))
            loaders["infer"] = infer_loader

        return loaders
