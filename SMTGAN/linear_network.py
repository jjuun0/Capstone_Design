import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class LinearStyleMixing(nn.Module):
    def __init__(self, in_dim, mode='train', activation='relu'):
        super().__init__()
        self.layer1 = nn.Linear(in_features=in_dim, out_features=512)
        self.layer2 = nn.Linear(in_features=512, out_features=512)
        self.layer3 = nn.Linear(in_features=512, out_features=512)
        self.layer4 = nn.Linear(in_features=512, out_features=512)
        self.layer5 = nn.Linear(in_features=512, out_features=512)

        self.activation = _get_activation_fn(activation)


    def forward(self, latent, random):
        concat = torch.cat([latent, random], dim=1)
        print(concat.size())
        out1 = self.activation(self.layer1(concat))
        print(out1.size())
        out2 = self.activation(self.layer2(out1))
        print(out2.size())
        out3 = self.activation(self.layer3(out2))
        print(out3.size())
        out4 = self.activation(self.layer4(out3))
        print(out4.size())
        out5 = self.layer5(out4)
        print(out5.size())
        # return [out1, out2, out3, out4, out5]
        return out5


class LinearMixStyleMixing(nn.Module):
    def __init__(self, in_dim, mode='train', name=None):
        super().__init__()
        self.coarse = LinearStyleMixing(in_dim=in_dim)
        self.medium = LinearStyleMixing(in_dim=in_dim)
        self.fine = LinearStyleMixing(in_dim=in_dim)
        self.mode = mode
        self.load_weights(name)

    def load_weights(self, name=None):
        if self.mode == 'test':
            ckpt = torch.load(
                (f'/home/project/바탕화면/Capstone_Design/JoJoGAN/models/tf_linear_{name}.pt'),
                map_location='cpu')
            self.coarse.load_state_dict(get_keys(ckpt, 'coarse'),
                                                         strict=False)
            self.medium.load_state_dict(get_keys(ckpt, 'medium'),
                                                         strict=False)
            self.fine.load_state_dict(get_keys(ckpt, 'fine'), strict=False)

    def forward(self, latent, random):
        if self.mode == 'train':
            medium = self.medium(latent[4:8,:], random[4:8,:])
        fine = self.fine(latent[8:,:], random[8:,:])

        # attn_style = torch.cat([coarse, medium, fine], dim=0)
        if self.mode == 'train':
            attn_style = torch.cat([latent[:4,:], medium, fine], dim=0)
        elif self.mode == 'test':
            attn_style = torch.cat([latent[:4,:], latent[4:8,:], fine], dim=0)

        return attn_style



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "silu":
        return F.silu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


if __name__ == '__main__':
    # linear style mixing
    # model = LinearStyleMixing(1024)
    a = torch.rand(4, 512)
    b = torch.rand(4, 512)
    # out = model(a, b)
    # print(out.shape)

    model = LinearStyleMixing(in_dim=1024)
    out = model(a, b)
