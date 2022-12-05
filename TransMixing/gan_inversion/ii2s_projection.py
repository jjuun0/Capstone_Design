import torch
import numpy as np
import os

from ii2s.models.II2S import II2S
from ii2s.options.face_embed_options import FaceEmbedOptions


def projection(img, name):
    parser = FaceEmbedOptions().parser

    args = parser.parse_args()

    ii2s = II2S(args)

    ##### Option 1: input folder
    # final_latents = ii2s.invert_images(image_path=args.input_dir, output_dir=args.output_dir,
    #                                    return_latents=True, align_input=True, save_output=True)

    #### Option 2: image path
    # final_latents = ii2s.invert_images(image_path=img, output_dir='ii2s/latents',
    #                                     return_latents=True, save_output=True)
    latent_path = os.path.join('ii2s/latents', name + '.npy')
    if not os.path.exists(latent_path):
        final_latents = ii2s.get_latent(img)
        final_latents = final_latents.detach()
        latent = final_latents.cpu().numpy()
        np.save(latent_path, latent)
    else:
        np_load = np.load(latent_path)
        final_latents = torch.from_numpy(np_load).cuda()

    # ##### Option 3: image path list
    # final_latents = ii2s.invert_images(image_path=['input/28.jpg', 'input/90.jpg'], output_dir=args.output_dir,
    #                                    return_latents=True, align_input=True, save_output=True)

    return final_latents


if __name__ == "__main__":

    parser = FaceEmbedOptions().parser

    # I/O arguments
    parser.add_argument('--input_dir', type=str, default='input',
                        help='The directory of the images to be inverted. ')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='The directory to save the latent codes and inversion images.')
    parser.add_argument('--img_path', type=str, default='input/jun.png',
                        help='The path of the image to be inverted. ')

    args = parser.parse_args()

    # with open(os.path.join(args.output_dir, "args.json"), 'w') as f:
    #     json.dump(args.__dict__, f, indent=2)

    projection(args)