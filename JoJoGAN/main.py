# plt.rcParams['figure.dpi'] = 150

from torchvision.utils import save_image

import torch
torch.backends.cudnn.benchmark = True
from torchvision import transforms, utils
from util import *
from PIL import Image
import math
import random
import os

import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm
from model import *
from e4e_projection import projection as e4e_projection
from copy import deepcopy

import time
import cv2
# cap = cv2.VideoCapture(0)


latent_dim = 512
device = 'cuda'

# Load original generator
original_generator = Generator(1024, latent_dim, 8, 2).to(device)
ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
original_generator.load_state_dict(ckpt["g_ema"], strict=False)
mean_latent = original_generator.mean_latent(10000)

generator = deepcopy(original_generator)

transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

output_transform = transforms.Compose(
    [
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)




# aligns and crops face
# aligned_face = align_face(filepath)
#
# my_w = e4e_projection(aligned_face, name, device).unsqueeze(0)
#
# aligned_face.save(f'aligned_{filename.split(".")[0]}.png')

pretrained = 'disney'
preserve_color = True

if preserve_color:
    ckpt = f'{pretrained}_preserve_color.pt'
else:
    ckpt = f'{pretrained}.pt'


ckpt = torch.load(os.path.join('models', ckpt), map_location=lambda storage, loc: storage)
generator.load_state_dict(ckpt["g"], strict=False)

#@title Generate results
# n_sample =  5#@param {type:"number"}
# seed = 3000 #@param {type:"number"}

# torch.manual_seed(seed)
with torch.no_grad():
    generator.eval()
    # for i in range(18):
    filename = f'{24}.png'
# filepath = f'test_input/{filename}'
    filepath = f'opencv_frame_{filename}'

    name = strip_path_extension(filepath) + '.pt'

    aligned_face = align_face(filepath)  # shape(1024, 1024)

    my_w = e4e_projection(aligned_face, name, device).unsqueeze(0)  # aligned_face: PIL.Image

    # original_my_sample = original_generator(my_w, input_is_latent=True)
    # while True:
    #     ret, frame = cap.read()
    #     frame = generator(frame, input_is_latent=True)
    #     cv2.imshow('test', frame)
    #     k = cv2.waitKey(1)
    #     if k == 27:
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
    # z = torch.randn(n_sample, latent_dim, device=device)

    # original_sample = original_generator([z], truncation=0.7, truncation_latent=mean_latent)
    # sample = generator([z], truncation=0.7, truncation_latent=mean_latent)


    my_sample = generator(my_w, input_is_latent=True)  # my_w: tensor(1, 18, 512)

    save_image(utils.make_grid(my_sample, normalize=True, range=(-1, 1)), f'{filename.split(".")[0]}.png')

# display reference images
# if pretrained == 'arcane_multi':
#     style_path = f'style_images_aligned/arcane_jinx.png'
# elif pretrained == 'sketch_multi':
#     style_path = f'style_images_aligned/sketch.png'
# else:
#     style_path = f'style_images_aligned/{pretrained}.png'
#
# style_image = transform(Image.open(style_path)).unsqueeze(0).to(device)
# face = transform(aligned_face).unsqueeze(0).to(device)
#
#
# my_output = torch.cat([style_image, face, my_sample], 0)
# display_image(utils.make_grid(my_output, normalize=True, range=(-1, 1)), title='My sample')

print('finish')
# output = torch.cat([original_sample, sample], 0)
# display_image(utils.make_grid(output, normalize=True, range=(-1, 1), nrow=n_sample), title='Random samples')
