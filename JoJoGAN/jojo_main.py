import torch
from style_mixing import MixingTransformer

torch.backends.cudnn.benchmark = True
from util import *
from model import *
from restyle_projection import projection as restyle_projection

from pathlib import Path
from ffhq_align import image_align
from config_parser import ConfigParser

def predict(config):
    inference_config = config['inference']
    latent_dim = 512  # latent dimension
    device = 'cuda'
    ckpt_name = inference_config['ckpt_name']
    # filenames = ["messi.jpg", "son.png", "you.jpg", "lee.jpg", "park.jpeg", "dong.jpg", "jin.jpg"]
    # filenames = ["jolie.jpg", "su.jpg", "jeon.jpg", "iu.jpeg", "han.jpeg", "ann.jpg", "choi.jpeg", "je.jpeg", "na.png", "yu.jpeg"]
    filenames = inference_config['inputs']
    gan_inversion = config['gan_inversion']
    character_path = inference_config['character_name']

    if gan_inversion == 'e4e':
        from e4e_projection import projection as e4e_projection
    # elif gan_inversion == 'ii2s':
    #     from ii2s_projection import projection as ii2s_projection
    elif gan_inversion == 'restyle':
        from restyle_projection import projection as restyle_projection
    # elif gan_inversion == 'stytr':
    #     from style_transformer.inference import projection as stytr_projection

    # trained jojogan models load
    generator = Generator(1024, latent_dim, 8, 2).to(device)
    ckpt = torch.load(ckpt_name, map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g"], strict=False)
    model_name = ckpt_name.split('/')[-1].split('.')[0]

    for filename in filenames:
        filepath = f'test_input/{filename}'
        name = strip_path_extension(filepath)

        # align face
        aligned_face = image_align(filepath)  # PIL.Image.Image, shape(1024, 1024)
        print('finish align face')

        # gan inversion
        if gan_inversion == 'e4e':
            my_w = e4e_projection(aligned_face, name, device)
        # elif gan_inversion == 'ii2s':
        #     my_w = ii2s_projection(aligned_face, name.split('/')[-1]).unsqueeze(0)
        elif gan_inversion == 'restyle':
            my_w = restyle_projection(aligned_face)
        # elif gan_inversion == 'stytr':
        #     my_w = stytr_projection(aligned_face)

        print('finish gan inversion')

        # Transformer Fine step
        mix_mode = True
        if mix_mode:
            character = f'/home/project/바탕화면/Capstone_Design/JoJoGAN/style_images/disney/{character_path}'
            character_align = image_align(character)
            character_w = restyle_projection(character_align)
            character_name = character.split('/')[-1].split('.')[0]

            style_mixing_model = MixingTransformer(d_model=512, nhead=4, dim_feedforward=1024, mode='test', name=character_name).to(device)
            style_mixing_model.eval()

        with torch.no_grad():
            generator.eval()
            if mix_mode:
                my_w = style_mixing_model(my_w[0], character_w[0]).unsqueeze(0)

            my_sample = generator(my_w, input_is_latent=True)  # my_w: tensor(1, 18, 512)

            my_sample = my_sample.cpu()

            my_sample = 1 + my_sample
            my_sample /= 2

            my_sample = my_sample[0]
            my_sample = 255 * torch.clip(my_sample, min=0, max=1)
            my_sample = my_sample.byte()

            my_sample = my_sample.permute(1, 2, 0).detach().numpy()
            my_sample = Image.fromarray(my_sample, mode="RGB")

            out_path = Path(f'/home/project/바탕화면/new_disney_result/{filename.split(".")[0]}_{model_name}.png')
            my_sample.save(str(out_path))
            print(f'---finish {filename}---')
    print('reference finish')

def predict_characters(config):
    latent_dim = 512  # latent dimension
    device = 'cuda'  # gpu setting
    # filenames = ["messi.jpg", "son.png", "you.jpg", "lee.jpg", "park.jpeg", "dong.jpg", "jin.jpg"]
    filenames = ["jolie.jpg", "su.jpg", "jeon.jpg", "iu.jpeg", "han.jpeg", "ann.jpg", "choi.jpeg", "je.jpeg", "na.png", "yu.jpeg"]
    gan_inversion = config['gan_inversion']
    for character in os.listdir('/home/project/바탕화면/disney/woman'):
        character_path = character
        ckpt_name = f"models/g_middle_{character.split('.')[0]}.pt"


        if gan_inversion == 'e4e':
            from e4e_projection import projection as e4e_projection
        elif gan_inversion == 'ii2s':
            from ii2s_projection import projection as ii2s_projection
        elif gan_inversion == 'restyle':
            from restyle_projection import projection as restyle_projection
        elif gan_inversion == 'stytr':
            from style_transformer.inference import projection as stytr_projection

        # trained jojogan models load
        generator = Generator(1024, latent_dim, 8, 2).to(device)
        ckpt = torch.load(ckpt_name, map_location=lambda storage, loc: storage)
        generator.load_state_dict(ckpt["g"], strict=False)
        model_name = ckpt_name.split('/')[-1].split('.')[0]

        for filename in filenames:
            filepath = f'test_input/{filename}'
            name = strip_path_extension(filepath)

            # align face
            aligned_face = image_align(filepath)  # PIL.Image.Image, shape(1024, 1024)
            print('finish align face')

            # gan inversion
            if gan_inversion == 'e4e':
                my_w = e4e_projection(aligned_face, name, device)
            elif gan_inversion == 'ii2s':
                my_w = ii2s_projection(aligned_face, name.split('/')[-1]).unsqueeze(0)
            elif gan_inversion == 'restyle':
                my_w = restyle_projection(aligned_face)
            elif gan_inversion == 'stytr':
                my_w = stytr_projection(aligned_face)

            print('finish gan inversion')

            mix_mode = True
            if mix_mode:
                character = f'/home/project/바탕화면/disney/woman/{character_path}'
                character_align = image_align(character)
                character_w = restyle_projection(character_align)
                character_name = character.split('/')[-1].split('.')[0]

                style_mixing_model = MixingTransformer(d_model=512, nhead=4, dim_feedforward=1024, mode='test', name=character_name).to(device)
                style_mixing_model.eval()

            with torch.no_grad():
                generator.eval()
                if mix_mode:
                    my_w = style_mixing_model(my_w[0], character_w[0]).unsqueeze(0)

                my_sample = generator(my_w, input_is_latent=True)  # my_w: tensor(1, 18, 512)

                my_sample = my_sample.cpu()

                my_sample = 1 + my_sample
                my_sample /= 2

                my_sample = my_sample[0]
                my_sample = 255 * torch.clip(my_sample, min=0, max=1)
                my_sample = my_sample.byte()

                my_sample = my_sample.permute(1, 2, 0).detach().numpy()
                my_sample = Image.fromarray(my_sample, mode="RGB")

                out_path = Path(f'/home/project/바탕화면/middle_disney_result/woman/{filename.split(".")[0]}_{model_name}.png')
                my_sample.save(str(out_path))
                print(f'---finish {filename}---')
        print('reference finish')


def flask_inference(img_name, model_name, preserve_color_info='refer'):
    latent_dim = 512  # latent dimension
    device = 'cuda'  # gpu setting
    ckpt_name = model_name
    filenames = [img_name]

    # Load original generator
    generator = Generator(1024, latent_dim, 8, 2).to(device)
    ckpt = torch.load(os.path.join('demo_flask/static/models', ckpt_name), map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g"], strict=False)
    model_name = ckpt_name.split('.')[0]

    for filename in filenames:
        filepath = f'demo_flask/static/face_input_image/{filename}'
        name = strip_path_extension(filepath)
        name = name.split('/')[-1]

        folder_name = model_name.split("_")[0]
        os.makedirs(f'demo_flask/static/result/{folder_name}', exist_ok=True)
        # align face
        aligned_face = image_align(filepath)
        # gan inversion
        my_w = restyle_projection(aligned_face)

        character_w = torch.load(f'demo_flask/static/latent/{model_name}.pt')

        style_mixing_model = MixingTransformer(d_model=512, nhead=4, dim_feedforward=1024, mode='flask', name=model_name).to(device)
        style_mixing_model.eval()

        with torch.no_grad():
            generator.eval()
            for i in range(2):
                if i:
                    my_w = style_mixing_model(my_w[0], character_w[0]).unsqueeze(0)

                my_sample = generator(my_w, input_is_latent=True)  # my_w: tensor(1, 18, 512)

                my_sample = my_sample.cpu()

                my_sample = 1 + my_sample
                my_sample /= 2

                my_sample = my_sample[0]
                my_sample = 255 * torch.clip(my_sample, min=0, max=1)
                my_sample = my_sample.byte()

                my_sample = my_sample.permute(1, 2, 0).detach().numpy()
                my_sample = Image.fromarray(my_sample, mode="RGB")
                model_name = model_name.split("_")[0]
                out_path = Path(f'demo_flask/static/result/{model_name}/{name}_{i}.png')
                my_sample.save(str(out_path))

            toonify1 = Image.open(f'demo_flask/static/result/{model_name}/{name}_0.png')
            toonify2 = Image.open(f'demo_flask/static/result/{model_name}/{name}_1.png')

            trans = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
            refer_img = Image.open(f'demo_flask/static/refer_align/{model_name}.png')
            img_plot(trans(refer_img).unsqueeze(0), trans(aligned_face).unsqueeze(0),
                     trans(toonify1).unsqueeze(0), trans(toonify2).unsqueeze(0),
                     f'demo_flask/static/result/{model_name}/{name}_result.png')

    print('reference finish')


if __name__ == '__main__':
    config = ConfigParser('./config.json')
    # predict(config)
    predict_characters(config)