import torch

from mmaction.apis import init_recognizer, inference_recognizer

config_file = '/home/alalwani_umass_edu/scratch/mmaction2/configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '/home/alalwani_umass_edu/scratch/mmaction2/checkpoints/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb_20201111-a9c34b54.pth'

# assign the desired device.
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # build the model from a config file and a checkpoint file
model = init_recognizer(config_file, checkpoint_file, device=device)

# test a single video and show the result:
video = '/home/alalwani_umass_edu/scratch/mmaction2/data/gym/subactions/0LtLS9wROrk_E_002407_002435_A_0003_0005.mp4'
labels = 'tools/data/gym/label_map.txt'
results = inference_recognizer(model, video)

# show the results
labels = open('tools/data/gym/label_map.txt').readlines()
labels = [x.strip() for x in labels]
results = [(labels[k[0]], k[1]) for k in results]

print(f'The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])
