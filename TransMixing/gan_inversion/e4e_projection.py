import os
import sys
import numpy as np
import torch
import torchvision.transforms as transforms

from argparse import Namespace
from e4e.models.psp import pSp
from util import *
from PIL import Image


@ torch.no_grad()
def projection(img, name, device='cuda'):
    model_path = 'models/e4e_ffhq_encode.pt'
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    net = pSp(opts, device).eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    img = transform(img).unsqueeze(0).to(device)  # (256, 256)
    images, w_plus = net(img, randomize_noise=False, return_latents=True)
    # result_file = {}
    # result_file['latent'] = w_plus[0]  # tensor (18, 512)
    # torch.save(result_file, name)
    return w_plus
