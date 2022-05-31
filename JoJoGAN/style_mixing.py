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
    def __init__(self, in_dim, mode='train'):
        super().__init__()
        self.layer1 = nn.Linear(in_features=in_dim, out_features=512)
        self.layer2 = nn.Linear(in_features=512, out_features=512)
        self.layer3 = nn.Linear(in_features=512, out_features=512)
        self.layer4 = nn.Linear(in_features=512, out_features=512)
        self.layer5 = nn.Linear(in_features=512, out_features=512)


    def forward(self, latent, random):
        concat = torch.cat([latent, random], dim=1)
        out1 = self.layer1(concat)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        # return [out1, out2, out3, out4, out5]
        return out5


class TransformerStyleMixing(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward models
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 512, 1024
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 1024, 512

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # self.pos_embedding = nn.Embedding(18, 512)

        # self.load_weights()

    def load_weights(self):
        ckpt = torch.load(
            ('/home/project/바탕화면/Capstone_Design/JoJoGAN/style_transformer/checkpoints/style_transformer_ffhq.pt'),
            map_location='cpu')
        # self.load_state_dict(get_keys(ckpt, 'encoder.module.transformerlayer_coarse'), strict=False)
        # self.load_state_dict(get_keys(ckpt, 'encoder.module.transformerlayer_medium'), strict=False)
        # self.load_state_dict(get_keys(ckpt, 'encoder.module.transformerlayer_fine'), strict=False)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def pos_embed(self, src, pos):
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None, ):
        # self-attention
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)

        # multi-attention
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        # tgt2 = self.norm1(tgt)
        # q = k = self.with_pos_embed(tgt2, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class Transformer(nn.Module):
    def __init__(self, depth, d_model, nhead, dim_feedforward, dropout=0.1, normalize_before=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                MixingTransformer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
                # TransformerStyleMixing(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
            )

    def forward(self, latent, random):
        for attn in self.layers:
            # latent = attn(latent, random) + latent
            latent = attn(latent, random)
        return latent


class MixingTransformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, mode='train', name=None):
        super().__init__()
        self.transformerlayer_coarse = TransformerStyleMixing(d_model=d_model, nhead=nhead,
                                                              dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformerlayer_medium = TransformerStyleMixing(d_model=d_model, nhead=nhead,
                                                              dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformerlayer_fine = TransformerStyleMixing(d_model=d_model, nhead=nhead,
                                                            dim_feedforward=dim_feedforward, dropout=dropout)
        # self.transformerlayer_fine = TransformerStyleMixing(d_model=10, nhead=5,
        #                                                     dim_feedforward=dim_feedforward, dropout=dropout)
        self.mode = mode
        self.load_weights(name)

        # self.coarse_pos = nn.Embedding(4, 512)
        # self.medium_pos = nn.Embedding(4, 512)
        # self.fine_pos = nn.Embedding(10, 512)
        #
        # self.random_coarse_pos = nn.Embedding(4, 512)
        # self.random_medium_pos = nn.Embedding(4, 512)
        # self.random_fine_pos = nn.Embedding(10, 512)
        # self.coarse_pos = nn.Parameter(torch.randn([1, 4, d_model]))
        # self.medium_pos = nn.Parameter(torch.randn([1, 4, d_model]))
        # self.fine_pos = nn.Parameter(torch.randn([1, 10, d_model]))
        #
        # self.random_coarse_pos = nn.Parameter(torch.randn([1, 4, d_model]))
        # self.random_medium_pos = nn.Parameter(torch.randn([1, 4, d_model]))
        # self.random_fine_pos = nn.Parameter(torch.randn([1, 10, d_model]))

    def load_weights(self, name=None):
        if self.mode == 'train':
            ckpt = torch.load(
                ('/home/project/바탕화면/Capstone_Design/JoJoGAN/style_transformer/checkpoints/style_transformer_ffhq.pt'),
                map_location='cpu')
        elif self.mode == 'test':
            ckpt = torch.load(
                (f'/home/project/바탕화면/Capstone_Design/JoJoGAN/models/tf_multi_{name}.pt'),
                map_location='cpu')
        elif self.mode == 'flask':
            ckpt = torch.load((f'demo_flask/static/models/tf/{name}.pt'), map_location='cpu')
            # ckpt = torch.load(
            #     (f'/home/project/바탕화면/Capstone_Design/JoJoGAN/models/tf_test_{name}.pt'),
            #     map_location='cpu')
        # self.transformerlayer_coarse.load_state_dict(get_keys(ckpt, 'encoder.module.transformerlayer_coarse'),
        #                                              strict=False)
        # self.transformerlayer_medium.load_state_dict(get_keys(ckpt, 'encoder.module.transformerlayer_medium'),
        #                                              strict=False)
        # self.transformerlayer_fine.load_state_dict(get_keys(ckpt, 'encoder.module.transformerlayer_fine'), strict=False)


        self.transformerlayer_coarse.load_state_dict(get_keys(ckpt, 'transformerlayer_coarse'),
                                                     strict=False)
        self.transformerlayer_medium.load_state_dict(get_keys(ckpt, 'transformerlayer_medium'),
                                                     strict=False)
        self.transformerlayer_fine.load_state_dict(get_keys(ckpt, 'transformerlayer_fine'), strict=False)

    def forward(self, latent, random):
        # t1:(1,18,512)
        # coarse = self.transformerlayer_coarse(latent[:, :4], random[:, :4])
        # medium = self.transformerlayer_medium(latent[:, 4:8], random[:, 4:8])
        # fine = self.transformerlayer_fine(latent[:, 8:], random[:, 8:])
        # attn_style = torch.cat([coarse, medium, fine], dim=1)

        # t2: (18,512)
        # coarse = self.transformerlayer_coarse(latent[:4,:], random[:4,:])
        if self.mode == 'train':
            medium = self.transformerlayer_medium(latent[4:8,:], random[4:8,:])
        fine = self.transformerlayer_fine(latent[8:,:], random[8:,:])

        # attn_style = torch.cat([coarse, medium, fine], dim=0)
        if self.mode == 'train':
            attn_style = torch.cat([latent[:4,:], medium, fine], dim=0)
        elif self.mode == 'test' or self.mode == 'flask':
            attn_style = torch.cat([latent[:4,:], latent[4:8,:], fine], dim=0)

        # swap = [9, 11, 15, 16, 17]
        # fix_layer = self.transformerlayer_fine(latent[swap], random[swap])
        # latent[swap] = fix_layer
        # attn_style = latent


        # t3: (512, 18)
        # coarse = self.transformerlayer_coarse(latent[:,:4], random[:,:4])
        # medium = self.transformerlayer_medium(latent[:,4:8], random[:,4:8])
        # fine = self.transformerlayer_fine(latent[:,8:], random[:,8:])
        # attn_style = torch.cat([coarse, medium, fine], dim=1)

        # coarse = self.transformerlayer_coarse(latent, random)
        # medium = self.transformerlayer_medium(coarse, random)
        # fine = self.transformerlayer_fine(medium, random)
        # attn_style = fine

        # coarse = self.transformerlayer_coarse(latent, random)
        # medium = self.transformerlayer_medium(coarse, random)
        # fine = self.transformerlayer_fine(medium, random)
        # attn_style = fine
        # print(self.coarse_pos)
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
    a = torch.rand(1, 18, 512)
    b = torch.rand(1, 18, 512)
    # out = model(a, b)
    # print(out.shape)

    model = TransformerStyleMixing(d_model=512, nhead=4, dim_feedforward=1024)
