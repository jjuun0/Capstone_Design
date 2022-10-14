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
from losses import id_loss, contextual_loss
# from losses.contextual_loss.vgg import VGG19

# from demo_flask.app import train_percent

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

    # names = ['rapunzel2.jpg']
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

        # crop and align the face
        # style_aligned_path = os.path.join('style_images_aligned', f'{name}.png')
        # style_aligned = Image.open(style_aligned_path).convert('RGB')
        style_aligned = image_align(style_path)
        # style_aligned.save(f'demo_flask/static/refer_align/{save_name}.png')
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
            # latent = restyle_projection(style_aligned)
            # latent = ii2s_projection(style_aligned, name)
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

    # preserve_color = True  # True: input's color preserve, False: refer's color preserve
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

        # with torch.no_grad():
        #     fake_img = generator(in_latent, input_is_latent=True)

        # np.save("stylized_face.npy", my_sample)
        # my_sample = fake_img.cpu()
        #
        # my_sample = 1 + my_sample
        # my_sample /= 2
        #
        # my_sample = my_sample[0]
        # my_sample = 255 * torch.clip(my_sample, min=0, max=1)
        # my_sample = my_sample.byte()
        #
        # my_sample = my_sample.permute(1, 2, 0).detach().numpy()
        # my_sample = Image.fromarray(my_sample, mode="RGB")
        #
        # out_path = Path(f'/home/project/바탕화면/jojogan_input/random_jojogan_no_train/{idx:03d}.png')
        # my_sample.save(str(out_path))

        # fake_img_256 = F.adaptive_avg_pool2d(fake_img, 256)
        # fake_feats = vggloss(fake_img_256)
        # fake_styles = [F.adaptive_avg_pool2d(fake_feat, output_size=1) for fake_feat in fake_feats]

        with torch.no_grad():
            real_feat = discriminator(targets)
        fake_feat = discriminator(fake_img)
        # # perceptual loss
        perceptual_loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)]) / len(fake_feat)
        loss = perceptual_loss
        # # l2_loss = F.mse_loss(targets.detach(), fake_img)  # l2 loss
        # # loss += l2_loss
        #
        # # identity loss
        # identity_loss = loss_id(fake_img, targets.detach()) * 0.75
        # loss += identity_loss

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
        # yield [idx, num_iter]
        if idx % 30:
            print(f'perceptual_loss : {perceptual_loss}')
    #         # print(f'l2 loss : {l2_loss}')
    #         print(f'id loss : {identity_loss}')  # 0.00xx
    #         # print(f'sty_loss : {sty_loss}')  # 0.2xx
    #         # print(f'per_loss : {per_loss}')  #
    #
        g_optim.zero_grad()
        loss.backward()
        g_optim.step()

    # save generative model
    # torch.save({"g": generator.state_dict()}, f'demo_flask/static/models/{save_name}_{preserve_color_info}_{gan_inversion}_per_id.pt')
    torch.save({"g": generator.state_dict()}, f'models/{save_name}_{name}.pt')
    # print('training finish!!')
    #

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
        # style_aligned_path = os.path.join('style_images_aligned', f'{name}.png')
        # style_aligned = Image.open(style_aligned_path).convert('RGB')
        style_aligned = image_align(style_path)
        style_aligned.save(f'demo_flask/static/refer_align/{save_name}.png')
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
            # latent = restyle_projection(style_aligned)
            # latent = ii2s_projection(style_aligned, name)
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


    # Finetune StyleGAN
    # alpha = 1.0
    # alpha = 1 - alpha

    # preserve_color = True  # True: input's color preserve, False: refer's color preserve
    # preserve_color = True if preserve_color_info == 'input' else False  # True: input's color preserve, False: refer's color preserve
    # Number of finetuning steps.
    # Different style reference may require different iterations. Try 200~500 iterations.
    num_iter = 300

    # load discriminator for perceptual loss
    generator = Generator(1024, latent_dim, 8, 2).to(device)
    discriminator = Discriminator(1024, 2).eval().to(device)

    ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"], strict=False)
    discriminator.load_state_dict(ckpt["d"], strict=False)



    # g_optim = optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))

    # Which layers to swap for generating a family of plausible real images -> fake image
    # if preserve_color:
    #     id_swap = [9, 11, 15, 16, 17]
    # else:
    #     id_swap = list(range(7, generator.n_latent))

    # loss_id = id_loss.IDLoss().to(device).eval()

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
            # yield idx // 30


        g_optim.zero_grad()
        loss.backward()
        g_optim.step()

    torch.save({"g": generator.state_dict()}, f'demo_flask/static/models/{save_name}.pt')
    torch.save(style_mixing_model.state_dict(), f'demo_flask/static/models/tf/{save_name}.pt')

        # in_latent = latents.clone()
        # in_latent[:, id_swap] = alpha * latents[:, id_swap] + (1 - alpha) * mean_w[:, id_swap]

        # fake_img = generator(in_latent, input_is_latent=True)

        # fake_img_256 = F.adaptive_avg_pool2d(fake_img, 256)
        # fake_feats = vggloss(fake_img_256)
        # fake_styles = [F.adaptive_avg_pool2d(fake_feat, output_size=1) for fake_feat in fake_feats]

        # with torch.no_grad():
        #     real_feat = discriminator(targets)
        # fake_feat = discriminator(fake_img)
        # # perceptual loss
        # perceptual_loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)]) / len(fake_feat)
        # loss = perceptual_loss
        # # l2_loss = F.mse_loss(targets.detach(), fake_img)  # l2 loss
        # # loss += l2_loss
        #
        # # identity loss
        # identity_loss = loss_id(fake_img, targets.detach()) * 0.75
        # loss += identity_loss

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

    #     # loss = per_loss + sty_loss
    #     # yield [idx, num_iter]
    #     if idx % 30:
    #         print(f'perceptual_loss : {perceptual_loss}')
    #         # print(f'l2 loss : {l2_loss}')
    #         print(f'id loss : {identity_loss}')  # 0.00xx
    #         # print(f'sty_loss : {sty_loss}')  # 0.2xx
    #         # print(f'per_loss : {per_loss}')  #
    #
    #     g_optim.zero_grad()
    #     loss.backward()
    #     g_optim.step()
    #
    # # save generative model
    # torch.save({"g": generator.state_dict()}, f'demo_flask/static/models/{save_name}_{preserve_color_info}.pt')
    print('training finish!!')

if __name__ == '__main__':
    # flask_train('anna2.jpeg', 'input')

    # train(save_name='jojo', img_name='man4.jpg', preserve_color_info='input')
    # train(save_name='jojo', img_name='boy1.jpeg', preserve_color_info='input')
    # train(save_name='jojo', img_name='luca.jpg', preserve_color_info='input')

    # train(save_name='jojo', img_name='rapunzel4.jpg', preserve_color_info='input')
    # train(save_name='jojo', img_name='anna.jpg', preserve_color_info='input')
    # train(save_name='jojo', img_name='chris2.png', preserve_color_info='input')

    # man
    # train('jojo', 'agnarr.png')
    # train('jojo', 'agnarr2.png')
    # train('jojo', 'augustin.png')
    # train('jojo', 'augustin2.png')
    # train('jojo', 'bruno.png')
    # train('jojo', 'bruno2.png')
    # train('jojo', 'chiefTui.png')
    # train('jojo', 'chris.png')
    # train('jojo', 'chris2.png')
    # train('jojo', 'felix.png')
    # train('jojo', 'felix2.png')
    # train('jojo', 'hans.png')
    # train('jojo', 'hans2.png')
    # train('jojo', 'hiro.png')
    # train('jojo', 'hiro2.png')
    # train('jojo', 'maui.png')
    # train('jojo', 'maui2.png')
    # train('jojo', 'oken.png')
    # train('jojo', 'plin.png')
    # train('jojo', 'plin2.png')
    # train('jojo', 'ralph.png')
    # train('jojo', 'tadashi.png')
    # train('jojo', 'tadashi2.png')

    # woman
    # train('jojo', 'alma.png')
    # train('jojo', 'alma2.png')
    # train('jojo', 'anna.png')
    # train('jojo', 'anna2.png')
    # train('jojo', 'dolores.png')
    # train('jojo', 'dolores2.png')
    # train('jojo', 'elsa.png')
    # train('jojo', 'elsa2.png')
    # train('jojo', 'elsa3.png')
    # train('jojo', 'godel.png')
    # train('jojo', 'gothel.png')
    # train('jojo', 'honey.png')
    # train('jojo', 'honey2.png')
    # train('jojo', 'iduna.png')
    # train('jojo', 'iduna2.png')
    # train('jojo', 'issabella.png')
    # train('jojo', 'issabella2.png')
    # train('jojo', 'julieta.png')
    # train('jojo', 'julieta2.png')
    # train('jojo', 'julieta3.png')
    # train('jojo', 'kamilo.png')
    # train('jojo', 'kamilo3.png')
    # train('jojo', 'mirabel.png')
    # train('jojo', 'mirabel2.png')
    # train('jojo', 'moana.png')
    # train('jojo', 'moana2.png')
    # train('jojo', 'pepa.png')
    # train('jojo', 'pepa2.png')
    # train('jojo', 'rapunzel.png')
    # train('jojo', 'rapunzel2.png')
    # train('jojo', 'ruisa.png')
    # train('jojo', 'ruisa2.png')
    # train('jojo', 'sina.png')
    # train('jojo', 'talla.png')
    # train('jojo', 'tamago.png')
    # train('jojo', 'vanellope.png')
    train('jojo', 'vanellope2.png')