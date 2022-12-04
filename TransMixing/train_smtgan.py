import torch
torch.backends.cudnn.benchmark = True
from util import *
from PIL import Image
import os
from pathlib import Path

from torch import optim
from tqdm import tqdm
from model import *
from style_mixing import MixingTransformer

from ffhq_align import image_align
from losses import id_loss


def train(save_name, img_name, preserve_color_info='refer'):

    print(f'{img_name} train start')

    latent_dim = 512
    device = 'cuda:0'

    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    names = [img_name]
    gan_inversion = 'e4e'
    if gan_inversion == 'e4e':
        from e4e_projection import projection as e4e_projection
    elif gan_inversion == 'ii2s':
        from ii2s_projection import projection as ii2s_projection
    elif gan_inversion == 'restyle':
        from restyle_projection import projection as restyle_projection
    elif gan_inversion == 'stytr':
        from style_transformer.inference import projection as stytr_projection

    targets = []
    latents = []

    for name in names:
        style_path = os.path.join('style_images/disney', name)
        assert os.path.exists(style_path), f"{style_path} does not exist!"

        name = strip_path_extension(name)
        style_aligned = image_align(style_path)

        # GAN invert
        style_code_path = os.path.join('inversion_codes', f'{name}.pt')
        if not os.path.exists(style_code_path):
            if gan_inversion == 'e4e':
                latent = e4e_projection(style_aligned, style_code_path, device)[0]
            elif gan_inversion == 'ii2s':
                latent = ii2s_projection(style_aligned, name)
            elif gan_inversion == 'restyle':
                latent = restyle_projection(style_aligned)[0]
            elif gan_inversion == 'stytr':
                latent = stytr_projection(style_aligned)
        else:
            latent = torch.load(style_code_path)['latent']

        targets.append(transform(style_aligned).to(device))
        latents.append(latent.to(device))

    targets = torch.stack(targets, 0)
    latents = torch.stack(latents, 0)


    # Finetune StyleGAN
    alpha = 1.0
    alpha = 1 - alpha

    preserve_color = True if preserve_color_info == 'input' else False  # True: input's color preserve, False: refer's color preserve
    # Number of finetuning steps.
    # Different style reference may require different iterations. Try 200~500 iterations.
    num_iter = 300

    # load discriminator for perceptual loss
    generator = Generator(1024, latent_dim, 8, 2).to(device)
    discriminator = Discriminator(1024, 2).eval().to(device)

    ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"], strict=False)
    discriminator.load_state_dict(ckpt["d"], strict=False)

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

        fake_img = generator(in_latent, input_is_latent=True)

        with torch.no_grad():
            real_feat = discriminator(targets)
        fake_feat = discriminator(fake_img)

        # perceptual loss
        perceptual_loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)]) / len(fake_feat)
        loss = perceptual_loss

        if idx % 30:
            print(f'perceptual_loss : {perceptual_loss}')

        g_optim.zero_grad()
        loss.backward()
        g_optim.step()

    # save generative model
    torch.save({"g": generator.state_dict()}, f'models/{save_name}_{name}.pt')

def flask_train(save_name, img_name):

    latent_dim = 512
    device = 'cuda:0'

    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # names = ['rapunzel2.jpg']
    names = [img_name]
    gan_inversion = 'restyle'
    if gan_inversion == 'e4e':
        from e4e_projection import projection as e4e_projection
    elif gan_inversion == 'ii2s':
        from ii2s_projection import projection as ii2s_projection
    elif gan_inversion == 'restyle':
        from restyle_projection import projection as restyle_projection
    elif gan_inversion == 'stytr':
        from style_transformer.inference import projection as stytr_projection

    targets = []
    latents = []

    for name in names:
        style_path = os.path.join('style_images/disney', name)
        assert os.path.exists(style_path), f"{style_path} does not exist!"

        name = strip_path_extension(name)

        # crop and align the face
        style_aligned = image_align(style_path)
        style_aligned.save(f'demo_flask/static/refer_align/{save_name}.png')

        # GAN invert
        style_code_path = os.path.join('inversion_codes', f'{name}.pt')
        if not os.path.exists(style_code_path):
            if gan_inversion == 'e4e':
                latent = e4e_projection(style_aligned, style_code_path, device)
            elif gan_inversion == 'ii2s':
                latent = ii2s_projection(style_aligned, name)
            elif gan_inversion == 'restyle':
                latent = restyle_projection(style_aligned)
            elif gan_inversion == 'stytr':
                latent = stytr_projection(style_aligned)
        else:
            latent = torch.load(style_code_path)['latent']

        targets.append(transform(style_aligned).to(device))
        latents.append(latent.to(device))
        torch.save(latent, f'demo_flask/static/latent/{save_name}.pt')
    targets = torch.stack(targets, 0)
    latents = torch.stack(latents, 0)


    # Number of finetuning steps.
    # Different style reference may require different iterations. Try 200~500 iterations.
    num_iter = 300

    # load discriminator for perceptual loss
    generator = Generator(1024, latent_dim, 8, 2).to(device)
    discriminator = Discriminator(1024, 2).eval().to(device)

    ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"], strict=False)
    discriminator.load_state_dict(ckpt["d"], strict=False)

    style_mixing_model = MixingTransformer(d_model=512, nhead=4, dim_feedforward=1024).to(device)
    style_mixing_model.train()
    params = list(generator.parameters()) + list(style_mixing_model.parameters())
    g_optim = optim.Adam(params, lr=2e-3, betas=(0, 0.99))

    for idx in tqdm(range(num_iter)):
        random_w = generator.get_latent(torch.randn([latents.size(0), latent_dim]).to(device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
        mix_w = style_mixing_model(latent[0], random_w[0]).unsqueeze(0)
        fake_img = generator(mix_w, input_is_latent=True)

        with torch.no_grad():
            real_feat = discriminator(targets)
        fake_feat = discriminator(fake_img)

        perceptual_loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)]) / len(fake_feat)
        loss = perceptual_loss

        if idx % 30:
            print(f'perceptual_loss : {perceptual_loss}')

        g_optim.zero_grad()
        loss.backward()
        g_optim.step()

    torch.save({"g": generator.state_dict()}, f'demo_flask/static/models/{save_name}.pt')
    torch.save(style_mixing_model.state_dict(), f'demo_flask/static/models/tf/{save_name}.pt')

    print('training finish!!')

if __name__ == '__main__':
    train('test', 'vanellope2.png')