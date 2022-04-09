import cv2
import json
import os
import time
from tqdm import tqdm
import sys

video_list = sys.argv[1:]
print(len(video_list))
print(video_list)
validation_set = set()

with open('../../../data/gym/annotations/gym99_val_org.txt') as f:
	lines = f.readlines()
	for line in lines:
		validation_set.add(line.split(' ')[0])

f = open('../../../data/gym/annotations/annotation.json')
annotations = json.load(f)

target_video_set = set(os.listdir('../../../data/gym/subaction_frames_full_validation_set'))

for video in tqdm(video_list):
	print(video)
	if not os.path.exists('../../../data/gym/videos/'+video+'.mp4'):
		continue
	cap = cv2.VideoCapture('../../../data/gym/videos/'+video+'.mp4')
	fps = cap.get(cv2.CAP_PROP_FPS)
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if annotations[video].items is not None:
		for k2,v2 in annotations[video].items():
			if v2['segments'] is not None:
				for k3,v3 in v2['segments'].items():
					if video+'_'+str(k2)+'_'+str(k3) in target_video_set or video+'_'+str(k2)+'_'+str(k3) not in validation_set:
						continue
					os.mkdir('../../../data/gym/subaction_frames_full_validation_set/'+video+'_'+str(k2)+'_'+str(k3))
					absolute_start_frame = (v2['timestamps'][0][0]+v3['timestamps'][0][0])*fps
					absolute_end_frame = (v2['timestamps'][0][0]+v3['timestamps'][-1][1])*fps
					cap.set(1,int(absolute_start_frame+1))
					count = absolute_start_frame
					ret = True
					count_file = 0
					while ret and count < absolute_end_frame:
						ret, frame = cap.read()
						cv2.imwrite('../../../data/gym/subaction_frames_full_validation_set/'+video+'_'+str(k2)+'_'+str(k3)+'/'+'img_'+str(count_file).zfill(5)+'.jpg',frame)
						count+=1
						count_file+=1
