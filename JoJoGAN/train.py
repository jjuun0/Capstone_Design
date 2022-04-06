import torch
torch.backends.cudnn.benchmark = True
from util import *
from PIL import Image
import os

from torch import optim
from tqdm import tqdm
from model import *
from e4e_projection import projection as e4e_projection


latent_dim = 512
device = 'cuda:0'

transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

names = ['ziwoo.jpeg'] #@param {type:"raw"}

targets = []
latents = []

for name in names:
    style_path = os.path.join('style_images', name)
    assert os.path.exists(style_path), f"{style_path} does not exist!"

    name = strip_path_extension(name)

    # crop and align the face
    style_aligned_path = os.path.join('style_images_aligned', f'{name}.png')
    if not os.path.exists(style_aligned_path):
        style_aligned = align_face(style_path)
        style_aligned.save(style_aligned_path)
    else:
        style_aligned = Image.open(style_aligned_path).convert('RGB')

    # GAN invert
    style_code_path = os.path.join('inversion_codes', f'{name}.pt')
    if not os.path.exists(style_code_path):
        latent = e4e_projection(style_aligned, style_code_path, device)
    else:
        latent = torch.load(style_code_path)['latent']

    targets.append(transform(style_aligned).to(device))
    latents.append(latent.to(device))

targets = torch.stack(targets, 0)
latents = torch.stack(latents, 0)


# Finetune StyleGAN
alpha = 1.0  # @param {type:"slider", min:0, max:1, step:0.1}
alpha = 1 - alpha

preserve_color = True  # True: input's color preserve, False: refer's color preserve
# Number of finetuning steps.
# Different style reference may require different iterations. Try 200~500 iterations.
num_iter = 300

# load discriminator for perceptual loss
generator = Generator(1024, latent_dim, 8, 2).to(device)
discriminator = Discriminator(1024, 2).eval().to(device)

ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
generator.load_state_dict(ckpt["g_ema"], strict=False)
discriminator.load_state_dict(ckpt["d"], strict=False)
# mean_latent = original_generator.mean_latent(10000)

g_optim = optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))

# Which layers to swap for generating a family of plausible real images -> fake image
if preserve_color:
    id_swap = [9, 11, 15, 16, 17]
else:
    id_swap = list(range(7, generator.n_latent))

for idx in tqdm(range(num_iter)):
    mean_w = generator.get_latent(torch.randn([latents.size(0), latent_dim]).to(device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
    in_latent = latents.clone()
    in_latent[:, id_swap] = alpha * latents[:, id_swap] + (1 - alpha) * mean_w[:, id_swap]

    img = generator(in_latent, input_is_latent=True)

    with torch.no_grad():
        real_feat = discriminator(targets)
    fake_feat = discriminator(img)

    loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)]) / len(fake_feat)

    g_optim.zero_grad()
    loss.backward()
    g_optim.step()

# save generative model
torch.save({"g": generator.state_dict()}, f'models/{name}.pt')
print('training finish!!')