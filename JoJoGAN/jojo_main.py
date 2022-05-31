import torch
from style_mixing import MixingTransformer
from linear_network import LinearMixStyleMixing

torch.backends.cudnn.benchmark = True
from img2gif import to_gif
from util import *
from model import *


from pathlib import Path
from ffhq_align import image_align
from config_parser import ConfigParser

def predict(config):
    inference_config = config['inference']

    latent_dim = 512  # latent dimension
    device = 'cuda'  # gpu setting
    ckpt_name = inference_config['ckpt_name']
    # filenames = ["messi.jpg", "son.png", "you.jpg", "lee.jpg", "park.jpeg", "dong.jpg", "jin.jpg"]
    # filenames = ["jolie.jpg", "su.jpg", "jeon.jpg", "iu.jpeg", "han.jpeg", "ann.jpg", "choi.jpeg", "je.jpeg", "na.png", "yu.jpeg"]
    filenames = inference_config['inputs']
    gan_inversion = config['gan_inversion']
    character_path = inference_config['character_name']

    if gan_inversion == 'e4e':
        from e4e_projection import projection as e4e_projection
    elif gan_inversion == 'ii2s':
        from ii2s_projection import projection as ii2s_projection
    elif gan_inversion == 'restyle':
        from restyle_projection import projection as restyle_projection
    elif gan_inversion == 'stytr':
        from style_transformer.inference import projection as stytr_projection

    # Load original generator
    # generator = Generator(1024, latent_dim, 8, 2).to(device)
    # ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    # generator.load_state_dict(ckpt["g_ema"], strict=False)  # g_ema : average of the weights of generator


    # trained jojogan models load
    generator = Generator(1024, latent_dim, 8, 2).to(device)
    ckpt = torch.load(ckpt_name, map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g"], strict=False)
    # model_name = ckpt_name.split('/')[-1].split('.')[0].split('_')[-1]
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

        # torch.save(my_w, f'{name}.pt')
        print('finish gan inversion')

        mix_mode = False
        if mix_mode:
            character = f'/home/project/바탕화면/Capstone_Design/JoJoGAN/style_images/disney/{character_path}'
            character_align = image_align(character)
            character_w = restyle_projection(character_align)
            character_name = character.split('/')[-1].split('.')[0]

            # style_mixing_model = LinearMixStyleMixing(1024, mode='test', name=character_name).to(device)
            style_mixing_model = MixingTransformer(d_model=512, nhead=4, dim_feedforward=1024, mode='test', name=character_name).to(device)
            style_mixing_model.eval()

        with torch.no_grad():
            generator.eval()
            if mix_mode:
                my_w = style_mixing_model(my_w[0], character_w[0]).unsqueeze(0)

            # face input image
            # filename = 'iu.jpeg'
            # filepath = f'test_input/{filename}'
            #
            # name = strip_path_extension(filepath)
            # name = name.split('/')[-1]

            # align face
            # aligned_face = align_face(filepath)  # shape(1024, 1024)
            # aligned_face = image_align(filepath)  # shape(1024, 1024)
            # print('aligned image finish')

            # gan inversion
            # my_w = e4e_projection(aligned_face, name, device).unsqueeze(0)  # aligned_face: PIL.Image
            # my_w = restyle_projection(aligned_face).unsqueeze(0)
            # my_w = ii2s_projection(aligned_face, name).unsqueeze(0)
            # print('got latent vector')

            # latent vector -> applying style(happy, young, sad)
            # latent_name = 'age'
            # expression = np.load(f'models/latent/{latent_name}.npy')
            # expression2 = np.load(f'models/latent/emotion_happy.npy')
            # alpha = 8
            # for alpha in range(-10, 11, 5):
            #     w_new = my_w + alpha * torch.tensor(expression, dtype=torch.float32).unsqueeze(0).cuda() \
            #             + 5 * torch.tensor(expression2, dtype=torch.float32).unsqueeze(0).cuda()

            my_sample = generator(my_w, input_is_latent=True)  # my_w: tensor(1, 18, 512)

            my_sample = my_sample.cpu()

            my_sample = 1 + my_sample
            my_sample /= 2

            my_sample = my_sample[0]
            my_sample = 255 * torch.clip(my_sample, min=0, max=1)
            my_sample = my_sample.byte()

            my_sample = my_sample.permute(1, 2, 0).detach().numpy()
            my_sample = Image.fromarray(my_sample, mode="RGB")

            # out_path = Path(f'/home/project/바탕화면/jojogan_input/final_result/final/{filename.split(".")[0]}_{model_name}.png')
            out_path = Path(f'/home/project/바탕화면/jojogan_input/jojo/e4e/{filename.split(".")[0]}_{model_name}.png')
            my_sample.save(str(out_path))
            print(f'---finish {filename}---')
    print('reference finish')


def flask_inference(img_name, model_name, preserve_color_info='refer'):
    latent_dim = 512  # latent dimension
    device = 'cuda'  # gpu setting
    ckpt_name = model_name
    # filenames =
    # filenames = ["jolie.jpg", "su.jpg", "jeon.jpg", "iu.jpeg", "han.jpeg"]
    filenames = [img_name]
    # gan_inversion= config['gan_inversion']
    # gan_inversion = 'restyle'
    from restyle_projection import projection as restyle_projection
    # if gan_inversion == 'e4e':
    #     from e4e_projection import projection as e4e_projection
    # elif gan_inversion == 'ii2s':
    #     from ii2s_projection import projection as ii2s_projection
    # elif gan_inversion == 'restyle':
    #     from restyle_projection import projection as restyle_projection
    # elif gan_inversion == 'stytr':
    #     from style_transformer.inference import projection as stytr_projection

    # Load original generator
    generator = Generator(1024, latent_dim, 8, 2).to(device)
    ckpt = torch.load(os.path.join('demo_flask/static/models', ckpt_name), map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g"], strict=False)
    model_name = ckpt_name.split('.')[0]

    for filename in filenames:
        # filename = 'chris_hemsworth.jpeg'

        filepath = f'demo_flask/static/face_input_image/{filename}'
        name = strip_path_extension(filepath)
        name = name.split('/')[-1]
        # if os.path.exists(f'demo_flask/static/result/{filename.split(".")[0]}'):
        #     shutil.rmtree(f'demo_flask/static/result/{filename.split(".")[0]}')

        # os.makedirs(f'demo_flask/static/result/{filename.split(".")[0]}', exist_ok=True)
        folder_name = model_name.split("_")[0]
        os.makedirs(f'demo_flask/static/result/{folder_name}', exist_ok=True)
        # align face
        aligned_face = image_align(filepath)  # PIL.Image.Image, shape(1024, 1024)
        # gan inversion
        # my_w = e4e_projection(aligned_face, name, device).unsqueeze(0)  # aligned_face: PIL.Image
        # my_w = restyle_projection(aligned_face).unsqueeze(0)
        # my_w = ii2s_projection(aligned_face, name).unsqueeze(0)
        # my_w = stytr_projection(aligned_face).unsqueeze(0)
        my_w = restyle_projection(aligned_face)
        # if gan_inversion == 'e4e':
        #     my_w = e4e_projection(aligned_face, name, device)
        # elif gan_inversion == 'ii2s':
        #     my_w = ii2s_projection(aligned_face, name).unsqueeze(0)
        # elif gan_inversion == 'restyle':
        #     my_w = restyle_projection(aligned_face)
        # elif gan_inversion == 'stytr':
        #     my_w = stytr_projection(aligned_face).unsqueeze(0)

        # mix_mode = True if preserve_color_info == 'reference' else False
        # if mix_mode:
            # character = f'/home/project/바탕화면/Capstone_Design/JoJoGAN/style_images/disney/{character_path}'
            # character_align = image_align(character)
            # character_w = restyle_projection(character_align)
        character_w = torch.load(f'demo_flask/static/latent/{model_name}.pt')
        # character_name = character.split('/')[-1].split('.')[0]

        # style_mixing_model = LinearMixStyleMixing(1024, mode='test', name=character_name).to(device)
        style_mixing_model = MixingTransformer(d_model=512, nhead=4, dim_feedforward=1024, mode='flask', name=model_name).to(device)
        style_mixing_model.eval()

        with torch.no_grad():
            generator.eval()
            for i in range(2):
                # if mix_mode:
                #     my_w = style_mixing_model(my_w[0], character_w[0]).unsqueeze(0)
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

                # save_image(utils.make_grid(my_sample, normalize=True, value_range=(-1, 1)),
                #        f'/home/project/바탕화면/test img/{filename.split(".")[0]}_{latent_name}_{alpha}.png')
    print('reference finish')
    # return my_sample

def apply_attribute(w, generator, folder):
    # age, angry, disgust, easy, fear, happy, sad, surprise
    # happy

    # os.makedirs('tmp', exist_ok=True)
    happy = np.load(f'models/latent/emotion_happy.npy')
    # angry = np.load(f'models/latent/emotion_angry.npy')
    surprise = np.load(f'models/latent/emotion_surprise.npy')
    disgust = np.load(f'models/latent/emotion_disgust.npy')
    # fear = np.load(f'models/latent/emotion_fear.npy')
    # sad = np.load(f'models/latent/emotion_sad.npy')
    emotion_dict = {'happy': happy, 'surprise': surprise, 'disgust': disgust}
    for emotion_name, latent in emotion_dict.items():
        for alpha in range(-10, 11, 1):
            w_new = w + alpha * torch.tensor(latent, dtype=torch.float32).unsqueeze(0).cuda()
            my_sample = generator(w_new, input_is_latent=True)  # my_w: tensor(1, 18, 512)

            my_sample = my_sample.cpu()
            # np.save("stylized_face.npy", my_sample)

            my_sample = 1 + my_sample
            my_sample /= 2

            my_sample = my_sample[0]
            my_sample = 255 * torch.clip(my_sample, min=0, max=1)
            my_sample = my_sample.byte()

            my_sample = my_sample.permute(1, 2, 0).detach().numpy()
            my_sample = Image.fromarray(my_sample, mode="RGB")
            os.makedirs(f'demo_flask/static/result/{folder}/{emotion_name}', exist_ok=True)
            out_path = Path(f'demo_flask/static/result/{folder}/{emotion_name}/{alpha+10:03d}.png')
            my_sample.save(str(out_path))

        to_gif(f'demo_flask/static/result/{folder}/{emotion_name}')
        print(f'finish {emotion_name} processing')


if __name__ == '__main__':
    config = ConfigParser('./config.json')
    predict(config)