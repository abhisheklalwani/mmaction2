# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import subprocess
import tqdm

import mmcv
import sys

video_list = sys.argv[1:]
print(video_list)

data_root = '../../../data/gym'
video_root = f'{data_root}/videos'
anno_root = f'{data_root}/annotations'
anno_file = f'{anno_root}/annotation.json'

event_anno_file = f'{anno_root}/event_annotation.json'
event_root = f'{data_root}/events'

videos = os.listdir(video_root)
videos = set(videos)
annotation = mmcv.load(anno_file)
event_annotation = {}

mmcv.mkdir_or_exist(event_root)

events = os.listdir(event_root)
events = set(events)

for k, v in tqdm.tqdm(annotation.items()):
    if k + '.mp4' not in videos:
        print(f'video {k} has not been downloaded')
        continue
    if k not in video_list:
        continue

    video_path = osp.join(video_root, k + '.mp4')
    for event_id, event_anno in tqdm.tqdm(v.items()):
        timestamps = event_anno['timestamps'][0]
        start_time, end_time = timestamps
        event_name = k + '_' + event_id

        output_filename = event_name + '.mp4'
        if output_filename in events:
            if os.path.getsize(event_root+'/'+output_filename) > 2000:
                continue
            else:
                os.remove(event_root+'/'+output_filename)
        command = [
            'ffmpeg', '-i',
            '"%s"' % video_path, '-ss',
            str(start_time), '-t',
            str(end_time - start_time), '-c:v', 'libx264', '-c:a', 'copy',
            '-threads', '16', '-loglevel', 'warning',
            '"%s"' % osp.join(event_root, output_filename)
        ]
        command = ' '.join(command)
        try:
            subprocess.check_output(
                command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print(
                f'Trimming of the Event {event_name} of Video {k} Failed',
                flush=True)

        segments = event_anno['segments']
        if segments is not None:
            event_annotation[event_name] = segments

mmcv.dump(event_annotation, event_anno_file)
