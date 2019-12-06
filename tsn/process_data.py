import argparse
import os
import pandas as pd
import cv2
from catalyst.utils import (
    set_global_seed, boolean_flag, get_pool, tqdm_parallel_imap
)


set_global_seed(42)
os.environ["OMP_NUM_THREADS"] = "1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        type=str,
        default=None)
    parser.add_argument(
        "--in-csv",
        type=str,
        required=True)
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True)
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True)
    parser.add_argument(
        "--n-cpu",
        type=int,
        default=None)
    boolean_flag(parser, "verbose", default=False)

    args = parser.parse_args()

    return args


def process_video(filepath, out_dir, datapath=None):
    datapath = datapath or ""
    # remove extension and class name form path
    localpath = filepath.rsplit(".", 1)[0]
    respath = f"{out_dir}/{localpath}/"
    os.makedirs(respath, exist_ok=True)

    capture = cv2.VideoCapture(os.path.join(datapath, filepath))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(frame_count):
        ret, frame = capture.read()
        if frame is None:
            continue
        cv2.imwrite("{}/{:06d}_{}.jpg".format(respath, i, fps), frame)
    capture.release()

    result = localpath
    return result


def csv2list(df):
    df = df.reset_index().drop("index", axis=1)
    df = list(df.to_dict("index").values())
    return df


def process_row(row):
    video = row["video"]
    videopath = process_video(
        row["filepath"], row["out_dir"],
        datapath=row.get("datapath", None))

    res = {
        "video": video,  # aka label
        "videopath": videopath,  # aka dir path
    }

    return res


def main(args):
    df = pd.read_csv(args.in_csv)
    if args.datapath:
        df["datapath"] = args.datapath

    if args.out_dir:
        df["out_dir"] = args.out_dir

    df_list = csv2list(df)
    with get_pool(args.n_cpu) as pool:
        df_list_out = tqdm_parallel_imap(process_row, df_list, pool)
    df_out = pd.DataFrame(df_list_out)
    df_out.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
