from torchvision.utils import save_image

import torch
torch.backends.cudnn.benchmark = True
from torchvision import utils
from util import *
import os

from model import *
from e4e_projection import projection as e4e_projection

latent_dim = 512
device = 'cuda'

# Load original generator
generator = Generator(1024, latent_dim, 8, 2).to(device)
ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
generator.load_state_dict(ckpt["g_ema"], strict=False)
# mean_latent = generator.mean_latent(10000)


transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# output_transform = transforms.Compose(
#     [
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ]
# )


# aligns and crops face
# aligned_face = align_face(filepath)
#
# my_w = e4e_projection(aligned_face, name, device).unsqueeze(0)
#
# aligned_face.save(f'aligned_{filename.split(".")[0]}.png')

# pretrained = 'disney'
# preserve_color = True
#
# if preserve_color:
#     ckpt = f'{pretrained}_preserve_color.pt'
# else:
#     ckpt = f'{pretrained}.pt'

ckpt = 'test.pt'


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
    filepath = f'/home/project/바탕화면/test img/opencv_frame_{filename}'

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


    my_sample = generator(my_w, input_is_latent=True)  # my_w: tensor(1, 18, 512)

    save_image(utils.make_grid(my_sample, normalize=True, value_range=(-1, 1)),
               f'/home/project/바탕화면/test img/{filename.split(".")[0]}.png')

print('reference finish')
