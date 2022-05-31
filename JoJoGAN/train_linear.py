import torch
torch.backends.cudnn.benchmark = True
from util import *
from PIL import Image
import os
from pathlib import Path

from linear_network import LinearMixStyleMixing
from torch import optim
from tqdm import tqdm
from model import *
from style_mixing import LinearStyleMixing, TransformerStyleMixing, Transformer, MixingTransformer
# from style_transformer_test import GradualStyleEncoder


from ffhq_align import image_align
from losses import id_loss, contextual_loss
from losses.contextual_loss.vgg import VGG19


def train(save_name, img_name, preserve_color_info='input'):

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

    targets = torch.stack(targets, 0)
    latents = torch.stack(latents, 0)
    latent = latent.to(device)


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

    loss_id = id_loss.IDLoss().to(device).eval()

    vggloss = VGG19().to(device).eval()
    loss_cl = contextual_loss.ContextualLoss(band_width=0.2)
    real_img_256 = F.adaptive_avg_pool2d(targets, 256).detach()
    real_feats = vggloss(real_img_256)
    real_styles = [F.adaptive_avg_pool2d(real_feat, output_size=1).detach() for real_feat in real_feats]

    PX_loss = 1.0
    ID_loss = 0.25
    CX_loss = 0.75
    Style_loss = 0.5
    # per_loss = torch.tensor(0.0).to(device)
    # vgg_weights = [0.0, 0.5, 1.0, 0.0, 0.0]
    # perc_loss = 1.0

    # style_mixing_model = LinearMixStyleMixing(1024).to(device)
    # style_mixing_model = TransformerStyleMixing(d_model=512, nhead=4, dim_feedforward=1024).to(device)
    # style_mixing_model = Transformer(depth=2, d_model=512, nhead=4, dim_feedforward=1024).to(device)
    style_mixing_model = MixingTransformer(d_model=512, nhead=4, dim_feedforward=1024).to(device)
    # style_mixing_model = MixingTransformer(d_model=4, nhead=4, dim_feedforward=64).to(device)
    style_mixing_model.train()
    params = list(generator.parameters()) + list(style_mixing_model.parameters())
    # g_optim = optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))
    g_optim = optim.Adam(params, lr=2e-3, betas=(0, 0.99))
    # style_mixing_model = GradualStyleEncoder(d_model=512, nhead=4, dim_feedforward=1024).to(device)
    # style_mixing_model = Transformer(depth=5, d_model=512, nhead=4, dim_feedforward=1024, normalize_before=True).to(device)

    for idx in tqdm(range(num_iter)):
        random_w = generator.get_latent(torch.randn([latents.size(0), latent_dim]).to(device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
        # mix_w = latent.clone()
        # in_latent[:, id_swap] = alpha * latents[:, id_swap] + (1 - alpha) * mean_w[:, id_swap]

        # linearStyleMixing
        # b = style_mixing_model(latent[:, id_swap], random_w[:, id_swap])
        # mix_w[:, id_swap] = b
        # mix_w[:, id_swap] = style_mixing_model(latent[:, id_swap], random_w[:, id_swap])
        # with torch.no_grad():
        #     mix_w = style_mixing_model(latent, random_w)  # latent: query, random_w: key, value:
        # with torch.no_grad():
        #     mix_w = style_mixing_model(latent, random_w)  # latent: query, random_w: key, value
        # t1
        # mix_w = style_mixing_model(latent, random_w)

        # mix_w = style_mixing_model(latent[0], random_w[0])
        # t2
        mix_w = style_mixing_model(latent[0], random_w[0]).unsqueeze(0)
        # t3
        # mix_w = style_mixing_model(latent[0].permute(1,0), random_w[0].permute(1,0)).unsqueeze(0).permute(0,2,1)
        # l2_loss = F.mse_loss(latent, mix_w)
        # loss = l2_loss


        fake_img = generator(mix_w, input_is_latent=True)
        # fake_img = generator(latent, input_is_latent=True)

        # with torch.no_grad():
        #     fake_img = generator(mix_w, input_is_latent=True)
        #
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
        # out_path = Path(f'/home/project/바탕화면/jojogan_input/tf_load/{idx:03d}.png')
        # my_sample.save(str(out_path))

        fake_img_256 = F.adaptive_avg_pool2d(fake_img, 256)
        fake_feats = vggloss(fake_img_256)
        fake_styles = [F.adaptive_avg_pool2d(fake_feat, output_size=1) for fake_feat in fake_feats]
        #
        with torch.no_grad():
            real_feat = discriminator(targets)
        fake_feat = discriminator(fake_img)

        # perceptual loss
        perceptual_loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)]) / len(fake_feat) * PX_loss
        loss = perceptual_loss
        # loss += perceptual_loss

        # l2 loss
        # l2_loss = F.mse_loss(targets.detach(), fake_img)
        # loss += l2_loss

        # identity loss
        # identity_loss = loss_id(fake_img, targets.detach()) * ID_loss
        # loss += identity_loss

        # dualstylegan - style loss
        # sty_loss = loss_cl(fake_feats[2], real_feats[2].detach()) * CX_loss
        # cx_loss = loss_cl(fake_feats[2], real_feats[2].detach()) * CX_loss
        # cx_loss = loss_cl(fake_feats[3], real_feats[3].detach()) * CX_loss
        # #
        # sty_loss = ((F.mse_loss(fake_styles[1], real_styles[1])
        #               + F.mse_loss(fake_styles[2], real_styles[2])) * Style_loss)
        # # #
        # loss += sty_loss
        # loss = cx_loss + sty_loss
        # loss += cx_loss

        # for ii, weight in enumerate(vgg_weights):
        #     if weight * perc_loss > 0:
        #         per_loss += F.l1_loss(fake_feats[ii], real_feats[ii].detach()) * weight * perc_loss

        # loss = per_loss + sty_loss
        # yield [idx, num_iter]
        if idx % 30:
            print(f'perceptual_loss : {perceptual_loss}')
            # print(f'l2 loss : {l2_loss}')
            # print(f'id loss : {identity_loss}')  # 0.00xx
            # print(f'sty_loss : {sty_loss}')  # 0.2xx
            # print(f'cx_loss : {cx_loss}')  # 0.2xx
        #     # print(f'per_loss : {per_loss}')  #
        #
        g_optim.zero_grad()
        loss.backward()
        g_optim.step()

    # save generative model
    # torch.save({"g": generator.state_dict()}, f'models/{save_name}_{preserve_color_info}_{gan_inversion}_per_id.pt')
    # torch.save({"g": generator.state_dict()}, f'models/cx_per05_loss_multi_transformer_2_relu_{name}.pt')
    # torch.save({"g": generator.state_dict()}, f'models/multi3_per1_cx025{name}.pt')
    # torch.save({"g": generator.state_dict()}, f'models/mixtf_t2_self_load_per_id_{name}.pt')
    torch.save({"g": generator.state_dict()}, f'models/g_multi_{name}.pt')
    torch.save(style_mixing_model.state_dict(), f'models/tf_multi_{name}.pt')

    # torch.save({"g": generator.state_dict()}, f'models/pretrained_gradual_{name}.pt')
    # print('training finish!!')

if __name__ == '__main__':
    # flask_train('anna2.jpeg', 'input')
    # train('test', 'anna2.jpeg')
    # train('test', 'rapunzel2.jpg')

    # train('test', 'moana.jpg')
    # train('test', 'sketch.jpeg')
    # train('test', 'piona.jpg')
    # train('test', 'oldman.jpeg')
    # train('test', 'encanto.jpg')

    # train('test', 'man4.jpg')
    # train('test', 'boy1.jpeg')
    # train('test', 'luca.jpg')

    # train('test', 'rapunzel4.jpg')
    # train('test', 'anna.jpg')
    # train('test', 'girl.png')

    # o
    # train('test', 'agnarr.png')
    # train('test', 'agnarr2.png')
    # train('test', 'augustin.png')
    # train('test', 'augustin2.png')
    # train('test', 'bruno.png')
    # train('test', 'bruno2.png')
    # train('test', 'chiefTui.png')
    # train('test', 'chris.png')
    # train('test', 'chris2.png')
    # train('test', 'felix.png')
    # train('test', 'felix2.png')
    # train('test', 'hans.png')
    # train('test', 'hans2.png')
    # train('test', 'hiro.png')
    # train('test', 'hiro2.png')
    # train('test', 'maui.png')
    # train('test', 'maui2.png')
    # train('test', 'oken.png')
    # train('test', 'plin.png')
    # train('test', 'plin2.png')
    # train('test', 'ralph.png')
    # train('test', 'tadashi.png')
    # train('test', 'tadashi2.png')

    # x
    train('test', 'alma.png')
    train('test', 'alma2.png')
    train('test', 'anna.png')
    train('test', 'anna2.png')
    train('test', 'dolores.png')
    train('test', 'dolores2.png')
    train('test', 'elsa.png')
    train('test', 'elsa2.png')
    train('test', 'elsa3.png')
    train('test', 'godel.png')
    train('test', 'gothel.png')
    train('test', 'honey.png')
    train('test', 'honey2.png')
    train('test', 'iduna.png')
    train('test', 'iduna2.png')
    train('test', 'issabella.png')
    train('test', 'issabella2.png')
    train('test', 'julieta.png')
    train('test', 'julieta2.png')
    train('test', 'julieta3.png')
    train('test', 'kamilo.png')
    train('test', 'kamilo3.png')
    train('test', 'mirabel.png')
    train('test', 'mirabel2.png')
    train('test', 'moana.png')
    train('test', 'moana2.png')
    train('test', 'pepa.png')
    train('test', 'pepa2.png')
    train('test', 'rapunzel.png')
    train('test', 'rapunzel2.png')
    train('test', 'ruisa.png')
    train('test', 'ruisa2.png')
    train('test', 'sina.png')
    train('test', 'talla.png')
    train('test', 'tamago.png')
    train('test', 'vanellope.png')
    train('test', 'vanellope2.png')


