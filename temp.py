import torch
from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
device = 'cpu'
# device = 'mps'
device = torch.device(device) # torch.device use cpu or gpu, if cpu, then device = 'cpu' else device = 'cuda:0'

model = init_recognizer(config_file, device=device)
# inference the demo video
ret = inference_recognizer(model, 'demo/demo.mp4', 'demo/label_map_k400.txt')
print (ret ) # [('sharpening knives', 43.114113), ('opening present', 41.971893), ('smoking', 38.595387), ('canoeing or kayaking', 37.236534), ('cleaning shoes', 37.221416)]
