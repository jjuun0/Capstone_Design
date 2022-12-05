import torch
import sys
import torchvision.transforms as transforms

sys.path.append(".")
sys.path.append("..")

from argparse import Namespace
from restyle.options.test_options import TestOptions
from restyle.models.psp import pSp
from restyle.models.e4e import e4e
from restyle.utils.model_utils import ENCODER_TYPES
from restyle.utils.inference_utils import run_on_batch, get_average_image


def projection(img):
    test_opts = TestOptions().parse()

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)

    if opts.encoder_type in ENCODER_TYPES['pSp']:
        net = pSp(opts)
    else:
        net = e4e(opts)

    net.eval()
    net.cuda()

    # get the image corresponding to the latent average
    avg_image = get_average_image(net, opts)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        input_cuda = img.cuda().float()
        latent = run_on_batch(input_cuda, net, opts, avg_image)
    return latent

if __name__ == '__main__':
    projection()
