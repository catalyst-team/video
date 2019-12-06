from typing import Dict
import json
import os
import numpy as np
import random
import cv2
import collections
import torch
import torch.nn as nn

from catalyst.data.augmentor import Augmentor
from catalyst.data.dataset import ListDataset
from catalyst.data.reader import ScalarReader, ReaderCompose, ImageReader
from catalyst.data.sampler import BalanceClassSampler
from catalyst.dl import ConfigExperiment
from catalyst.utils import imread
from catalyst.utils.pandas import read_csv_data


from albumentations import (
    Resize, JpegCompression, Normalize,
    HorizontalFlip, ShiftScaleRotate,
    HueSaturationValue, MotionBlur,
    RandomBrightnessContrast, BasicTransform,
    OpticalDistortion, CLAHE)


# ---- Augmentations ----
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
IMG_SIZE = 224


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


def preprocess_valid_data(data):
    # copy data for each second of video
    new_data = []
    for d in data:
        for start_interval in range(int(d["secs"])):
            d["offset"] = start_interval
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
        num_frames: int = None,
        num_segments: int = None,
        time_window:int = None,
        uniform_time_sample:bool = False,
        with_offset: bool = False
    ):
        """
        :param input_key: input key to use from annotation dict
        :param output_key: output key to use to store the result
        :param datapath: path to images dataset
            (so your can use relative paths in annotations)
        :param grayscale: boolean flag
            if you need to work only with grayscale images
        :param num_frames: number of frames to take from each segment
        :param num_segments: number of frames splitting
        :param time_window: length of sample crop from video  in seconds
        :param uniform_time_sample: use uniform frames sample from time interval or not
        :param with_offset: enable for sequential for second validation
        """
        self.input_key = input_key
        self.output_key = output_key
        self.datapath = datapath
        self.grayscale = grayscale
        self.num_frames = num_frames
        self.num_segments = num_segments
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
        fps = int(frames[0].split("/")[-1].split("_")[1].split(".")[0])

        # Step 1: sorting
        frames = sorted(frames)

        # Step 2: sampling by time
        if self.time_window:
            frame_window = self.time_window * fps
            if self.with_offset:
                start_frame = row["offset"] * fps
            elif len(frames) > frame_window:
                start_frame = np.random.randint(0, len(frames) - frame_window)
            else:
                start_frame = 0
            # mb less than frame_window
            frames = frames[start_frame: start_frame + frame_window]
        else:
            frame_window = len(frames)

        # Step 3: choosing frames from time range
        # Option a: uniform time sampling (with constant time step)
        if self.uniform_time_sample:
            frames_indexes = [
                int(frame_window / self.num_frames * i)
                for i in range(self.num_frames)
            ]
            # If frames less than needed - duplicate last.
            # Important to keep constant time step
            frames = [frames[min(i, len(frames) - 1)] for i in frames_indexes]

        # Option b: (self.num_frames is None) use all frames (bad idea)
        elif self.num_frames is not None:
            tmp_frames = []
            if self.num_segments is not None:
                # Option c: random n sample from each successive interval
                frames = np.array_split(frames, self.num_segments)
            else:
                # Option d: random n sample (at all) from all frames
                frames = [frames]

            for frames_ in frames:
                replace_ = self.num_frames > len(frames_)
                frames_ = np.random.choice(
                    frames_,
                    self.num_frames,
                    replace=replace_).tolist()
                tmp_frames.extend(frames_)

            tmp_frames = sorted(tmp_frames)  # because of random.choice
            frames = tmp_frames

        frames = [
            imread(x, rootpath=self.datapath, grayscale=self.grayscale)
            for x in frames
        ]
        result = {self.output_key: frames}
        return result


class Experiment(ConfigExperiment):
    def _prepare_logdir(self, config: Dict):
        model_params = config["model_params"]["tsn_model"]
        data_params = config["stages"]["data_params"]
        train_data_name = (
            data_params.get("in_csv_train")
                .split("/")[-1].split("_")[2].split(".")[0]
        )
        return f"fold_{train_data_name}" \
               f"-{data_params.get('uniform_time_sample')}" \
               f"-{data_params.get('num_frames')}" \
               f"-{data_params.get('num_segments')}" \
               f"-{model_params.get('early_consensus')}" \
               f"-{model_params.get('feature_net_skip_connection')}" \
               f"-{model_params.get('feature_net_hiddens')}" \
               f"-{','.join(model_params.get('consensus'))}" \
               f"-{model_params.get('kernel_size')}"

    def _postprocess_model_for_stage(self, stage: str, model: nn.Module, partial_bn=2, **kwargs):
        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        if stage in ["debug", "stage_head_train"]:
            for param in model_.encoder.parameters():
                param.requires_grad = False
        elif stage in ["stage_full_finetune", "stage_full_train"]:
            for param in model_.encoder.parameters():
                param.requires_grad = True

            count = 0
            for m in model_.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= partial_bn:
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        else:
            pass

        return model

    @staticmethod
    def get_transforms(stage: str = None, mode: str = None, input_size: int = 224):
        train_image_transforms = [
            OpticalDistortion(distort_limit=0.3, p=0.3),
            JpegCompression(quality_lower=50, p=0.8),
            HorizontalFlip(p=0.5),
            MotionBlur(p=0.5),
            ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=20,
                p=0.5),
            RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.2,
                p=0.4),
            HueSaturationValue(
                hue_shift_limit=3,
                sat_shift_limit=20,
                val_shift_limit=30,
                p=0.4),
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

    def get_datasets(
        self,
        stage: str,
        datapath: str = None,
        in_csv: str = None,
        in_csv_train: str = None,
        in_csv_valid: str = None,
        in_csv_infer: str = None,
        train_folds: str = None,
        valid_folds: str = None,
        tag2class: str = None,
        class_column: str = None,
        tag_column: str = None,
        folds_seed: int = 42,
        n_folds: int = 5,
        one_hot_classes: bool = None,
        num_frames: int = None,
        num_segments: int = None,
        time_window: int = None,
        uniform_time_sample: bool = False,
    ):
        datasets = collections.OrderedDict()
        tag2class = json.load(open(tag2class)) \
            if tag2class is not None \
            else None

        df, df_train, df_valid, df_infer = read_csv_data(
            in_csv=in_csv,
            in_csv_train=in_csv_train,
            in_csv_valid=in_csv_valid,
            in_csv_infer=in_csv_infer,
            train_folds=train_folds,
            valid_folds=valid_folds,
            tag2class=tag2class,
            class_column=class_column,
            tag_column=tag_column,
            seed=folds_seed,
            n_folds=n_folds
        )

        df_valid = preprocess_valid_data(df_valid)

        open_fn = [
            ScalarReader(
                input_key="class",
                output_key="targets",
                default_value=-1,
                dtype=np.int64
            )

        ]
        if one_hot_classes:
            open_fn.append(
                ScalarReader(
                    input_key="class",
                    output_key="targets_one_hot",
                    default_value=-1,
                    dtype=np.int64,
                    one_hot_classes=one_hot_classes
                )
            )

        open_fn_val = open_fn.copy()
        open_fn.append(
            VideoImageReader(
                input_key="filepath",
                output_key="features",
                datapath=datapath,
                num_frames=num_frames,
                num_segments=num_segments,
                time_window=time_window,
                uniform_time_sample=uniform_time_sample))
        open_fn_val.append(
            VideoImageReader(
                input_key="filepath",
                output_key="features",
                datapath=datapath,
                num_frames=num_frames,
                num_segments=num_segments,
                time_window=time_window,
                uniform_time_sample=uniform_time_sample,
                with_offset=True))

        open_fn = ReaderCompose(readers=open_fn)
        open_fn_val = ReaderCompose(readers=open_fn_val)

        for source, mode in zip(
                (df_train, df_valid, df_infer),
                ("train", "valid", "infer")):
            if len(source) > 0:
                dataset = ListDataset(
                    source,
                    open_fn=open_fn_val if mode == "valid" else open_fn,
                    dict_transform=self.get_transforms(
                        stage=stage, mode=mode
                    ),
                )
                dataset_dict = {"dataset": dataset}
                if mode == "train":
                    labels = [x["class"] for x in df_train]
                    sampler = BalanceClassSampler(labels, mode="upsampling")
                    dataset_dict['sampler'] = sampler

                datasets[mode] = dataset_dict
        return datasets
