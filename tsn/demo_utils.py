import cv2
import os
import datetime
import pandas as pd
import imageio
from model import tsn, ModelRunner
from data import DataSource
from annotation_parser import parse


def get_fps(vid_path):
    vid = imageio.get_reader(vid_path, 'ffmpeg')
    return round(vid.get_meta_data()['fps'])


def dataset_from_video(path, data_path='', offset_numerator=None, offset_denominator=None):
    # folder for one demo video
    os.makedirs(f'{path.split(".")[0]}', exist_ok=True)
    folders = []

    capture = cv2.VideoCapture(path)
    fps = get_fps(path)

    offset = None
    if offset_numerator is not None and offset_denominator is not None:
        offset = int(fps * offset_numerator / offset_denominator)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if offset is not None:
        frame_count -= offset
        for i in range(offset):
            _, _ = capture.read()

    for start_frame in range(0, frame_count, fps):
        if frame_count - start_frame < 20:  # skip last short sample
            break

        # folder for each second of demo video
        suffix = f'_{offset_numerator}' if offset is not None else ''
        sample_folder = f'{path.split(".")[0]}{suffix}/{start_frame}'
        print(f'Make folder {sample_folder}')
        os.makedirs(sample_folder, exist_ok=True)
        folders.append(sample_folder)

        for i in range(start_frame, start_frame + min(fps, frame_count - start_frame)):
            ret, frame = capture.read()
            if frame is None:
                continue
            cv2.imwrite('{}/{:06d}_{}.jpg'.format(sample_folder, i - start_frame, fps),
                        frame)

    capture.release()

    dataset = pd.DataFrame({'filepath': folders})
    dataset.filepath = dataset.filepath.apply(lambda x: os.path.join(data_path, x))
    return dataset


def prediction_to_label(probs):
    mapping = {
        0: 'walk',
        1: 'trot'
    }

    classes = probs.argmax(axis=1)
    res = []
    for x in classes:
        res.append(mapping.get(x, 'other'))
    return res


def prediction_to_video(pred, path, to_web=True):
    capture = cv2.VideoCapture(path)

    fps = get_fps(path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    codec = 'VP80' if to_web else 'MJPG'
    extension = 'webm' if to_web else 'mp4'
    fourcc = cv2.VideoWriter_fourcc(*codec)
    predicted_path = f'{path.split(".")[0]}_prediction.{extension}'

    out = cv2.VideoWriter(predicted_path, fourcc, fps,
                          (frameWidth, frameHeight))

    for i in range(frame_count):
        ret, frame = capture.read()
        if frame is None:
            continue

        text_index = i // fps
        text = pred[min(text_index, len(pred) - 1)]
        cv2.putText(frame, text=text, org=(30, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3, color=(0, 256, 0), lineType=cv2.LINE_AA, thickness=3)
        out.write(frame)

    capture.release()
    out.release()
    return predicted_path


def predict_config(weights, train_config, pred_config, path=None, out_prefix=None):
    model = tsn(train_config['model_params']['base_model'],
                train_config['model_params']['tsn_model'])

    if path is not None:
        pred_config['data_params']['in_csv_infer'] = path

    if out_prefix is not None:
        pred_config['args']['out_prefix'] = out_prefix

    datasource = DataSource()
    loaders = datasource.prepare_loaders(
        mode="infer",
        n_workers=pred_config['args']['workers'],
        batch_size=pred_config['args']['batch_size'],
        **pred_config['data_params']
    )

    runner = ModelRunner(model)
    callbacks = runner.prepare_callbacks(
            mode="infer",
            resume=weights,
            out_prefix=pred_config['args']['out_prefix'],
            **pred_config['callbacks_params']
        )
    runner.infer(loaders=loaders, callbacks=callbacks, verbose=True)


def create_reports(pred_df):
    def to_dt(x):
        return str(datetime.timedelta(seconds=x))

    def to_timedelta(d):
        return datetime.timedelta(hours=d.hour, minutes=d.minute, seconds=d.second)

    with_true_label = 'True label' in pred_df.columns

    # Intervals report
    predicted_intervals = \
        (~(pred_df['Predicted label'] == pred_df['Predicted label'].shift(1))).nonzero()[0].tolist()

    if with_true_label:
        true_intervals = (~(pred_df['True label'] == pred_df['True label'].shift(1))).nonzero()[
            0].tolist()
    else:
        true_intervals = []

    intervals = sorted(list(set(predicted_intervals + true_intervals + [int(
        pred_df.Second.max()) + 1])))

    intervals_df = []
    for i in range(0, len(intervals) - 1):
        row = {
            'Start Time': to_dt(intervals[i]),
            'End Time': to_dt(intervals[i + 1]),
            'Predicted label': pred_df['Predicted label'].iloc[intervals[i]],
            'Segment Duration': to_dt(intervals[i + 1] - intervals[i])
        }
        if with_true_label:
            row['True label'] = pred_df['True label'].iloc[intervals[i]]
        intervals_df.append(row)

    intervals_df = pd.DataFrame(intervals_df)
    intervals_col = ['Start Time', 'End Time', 'Segment Duration', 'Predicted label']
    if with_true_label:
        intervals_col += ['True label']
    intervals_df = intervals_df[intervals_col]

    # Summary report
    intervals_df_ = intervals_df.copy()
    intervals_df_['Segment Duration'] = pd.to_datetime(
        intervals_df_['Segment Duration']).dt.time.apply(to_timedelta)

    grouped = intervals_df_.groupby('Predicted label')
    grouped_df = pd.DataFrame({
        'Label': grouped['Segment Duration'].count().index.values,
        'Number of Segments': grouped['Segment Duration'].count().values,
        'Total Label Duration': grouped['Segment Duration'].sum().apply(lambda x: to_dt(x.seconds))
    })

    grouped_df = grouped_df.reset_index(drop=True)
    if with_true_label:
        accuracy = (pred_df['Predicted label'] == pred_df['True label']).mean()
        accuracy = int(accuracy * 100)
        grouped_df[f'Per second Accuracy: {accuracy}%'] = ''

    return intervals_df, grouped_df


def make_subs(df_intervals, filename='subs.vtt'):
    with open (filename, 'w') as f:
        f.write('WEBVTT\n\n')
        for row in df_intervals.values:
            f.write(f'0{row[0]}.000 --> 0{row[1]}.000\n{row[3]}\n')