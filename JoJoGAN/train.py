import torch
torch.backends.cudnn.benchmark = True
from util import *
from PIL import Image
import os

from torch import optim
from tqdm import tqdm
from model import *
# from e4e_projection import projection as e4e_projection
from restyle_projection import projection as restyle_projection
# from ii2s_projection import projection as ii2s_projection
from ffhq_align import image_align
from losses import id_loss, contextual_loss
from losses.contextual_loss.vgg import VGG19


latent_dim = 512
device = 'cuda:0'

transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

names = ['rapunzel2.jpg']

targets = []
latents = []

for name in names:
    style_path = os.path.join('style_images', name)
    assert os.path.exists(style_path), f"{style_path} does not exist!"

    name = strip_path_extension(name)

    # crop and align the face
    # style_aligned_path = os.path.join('style_images_aligned', f'{name}.png')
    # style_aligned = Image.open(style_aligned_path).convert('RGB')
    style_aligned = image_align(style_path)
    # if not os.path.exists(style_aligned_path):
    #     # style_aligned = align_face(style_path)
    #     style_aligned = image_align(style_path)
    #     # style_aligned.save(style_aligned_path)
    # else:
    #     style_aligned = Image.open(style_aligned_path).convert('RGB')

    # GAN invert
    style_code_path = os.path.join('inversion_codes', f'{name}.pt')
    if not os.path.exists(style_code_path):
        # latent = e4e_projection(style_aligned, style_code_path, device)
        latent = restyle_projection(style_aligned)
        # latent = ii2s_projection(style_aligned, name)
    else:
        latent = torch.load(style_code_path)['latent']

    targets.append(transform(style_aligned).to(device))
    latents.append(latent.to(device))

targets = torch.stack(targets, 0)
latents = torch.stack(latents, 0)


# Finetune StyleGAN
alpha = 1.0
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

g_optim = optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))

# Which layers to swap for generating a family of plausible real images -> fake image
if preserve_color:
    id_swap = [9, 11, 15, 16, 17]
else:
    id_swap = list(range(7, generator.n_latent))

loss_id = id_loss.IDLoss().to(device).eval()

# vggloss = VGG19().to(device).eval()
# loss_cl = contextual_loss.ContextualLoss(band_width=0.2)
# real_img_256 = F.adaptive_avg_pool2d(targets, 256).detach()
# real_feats = vggloss(real_img_256)
# real_styles = [F.adaptive_avg_pool2d(real_feat, output_size=1).detach() for real_feat in real_feats]

# CX_loss = 0.25
# Style_loss = 0.25
# per_loss = torch.tensor(0.0).to(device)
# vgg_weights = [0.0, 0.5, 1.0, 0.0, 0.0]
# perc_loss = 1.0
for idx in tqdm(range(num_iter)):
    mean_w = generator.get_latent(torch.randn([latents.size(0), latent_dim]).to(device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
    in_latent = latents.clone()
    in_latent[:, id_swap] = alpha * latents[:, id_swap] + (1 - alpha) * mean_w[:, id_swap]

    fake_img = generator(in_latent, input_is_latent=True)

    # fake_img_256 = F.adaptive_avg_pool2d(fake_img, 256)
    # fake_feats = vggloss(fake_img_256)
    # fake_styles = [F.adaptive_avg_pool2d(fake_feat, output_size=1) for fake_feat in fake_feats]

    with torch.no_grad():
        real_feat = discriminator(targets)
    fake_feat = discriminator(fake_img)
    # perceptual loss
    loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)]) / len(fake_feat)

    l2_loss = F.mse_loss(targets.detach(), fake_img)  # l2 loss
    loss += l2_loss

    # identity loss
    id_loss = loss_id(fake_img, targets.detach()) * 0.75
    loss += id_loss

    # dualstylegan - style loss
    # sty_loss = loss_cl(fake_feats[2], real_feats[2].detach()) * CX_loss
    #
    # sty_loss += ((F.mse_loss(fake_styles[1], real_styles[1])
    #               + F.mse_loss(fake_styles[2], real_styles[2])) * Style_loss)
    #
    # loss += sty_loss

    # for ii, weight in enumerate(vgg_weights):
    #     if weight * perc_loss > 0:
    #         per_loss += F.l1_loss(fake_feats[ii], real_feats[ii].detach()) * weight * perc_loss

    # loss = per_loss + sty_loss
    if idx % 30:
        print(f'l2 loss : {l2_loss}')
        print(f'id loss : {id_loss}')  # 0.00xx
        # print(f'sty_loss : {sty_loss}')  # 0.2xx
        # print(f'per_loss : {per_loss}')  #

    g_optim.zero_grad()
    loss.backward()
    g_optim.step()

# save generative model
torch.save({"g": generator.state_dict()}, f'models/{name}_restyle_per_l2_id.pt')
print('training finish!!')