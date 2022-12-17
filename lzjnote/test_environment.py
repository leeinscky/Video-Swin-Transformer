# https://github.com/leeinscky/Video-Swin-Transformer/blob/master/docs/install.md#verification

import torch
import sys
sys.path.append("/Users/lizejian/cambridge/mphil_project/learn/Video-Swin-Transformer") # Adds higher directory to python modules path.
# sys.path.append("..") # Adds higher directory to python modules path.
from mmaction.apis import init_recognizer, inference_recognizer

# config_file = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
# device = 'cuda:0' # or 'cpu'
config_file = '/Users/lizejian/cambridge/mphil_project/learn/Video-Swin-Transformer/configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
device = 'cpu'
device = torch.device(device)

model = init_recognizer(config_file, device=device)
# inference the demo video
# ret = inference_recognizer(model, '../demo/demo.mp4', '../demo/label_map_k400.txt')
ret = inference_recognizer(model, '/Users/lizejian/cambridge/mphil_project/learn/Video-Swin-Transformer/demo/demo.mp4', '/Users/lizejian/cambridge/mphil_project/learn/Video-Swin-Transformer/demo/label_map_k400.txt')
print (ret) # [('sharpening knives', 43.114113), ('opening present', 41.971893), ('smoking', 38.595387), ('canoeing or kayaking', 37.236534), ('cleaning shoes', 37.221416)]