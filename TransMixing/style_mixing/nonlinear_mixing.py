import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


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
                     query_pos: Optional[Tensor] = None, iter=None, step=None):
        # self-attention
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)

        # multi-attention

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt.unsqueeze(1), query_pos),
                                   key=self.with_pos_embed(memory.unsqueeze(1), pos),
                                   # value=tgt, attn_mask=memory_mask,
                                   value=memory.unsqueeze(1), attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # visualize attention score
        # if step:
        #     attention = tgt2.squeeze().detach().cpu().numpy()
        #     att_re = np.zeros(attention.shape)
        #     for i in range(attention.shape[0]):
        #         att_re[i] = attention[i] / attention[i].max()
        #     # tgt_cpu = tgt.detach().cpu().numpy()
        #     # attention_mix = tgt_cpu * att_re
        #     plt.imshow(attention, interpolation='nearest', aspect='auto')
        #     plt.savefig(f'moana_middle/{step}/{iter:03d}.png')

        tgt2 = tgt2.squeeze()
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
                query_pos: Optional[Tensor] = None, iter=None, step=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, iter=iter, step=step)


class MixingTransformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, mode='train', name=None):
        super().__init__()
        self.transformerlayer_coarse = TransformerStyleMixing(d_model=d_model, nhead=nhead,
                                                              dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformerlayer_medium = TransformerStyleMixing(d_model=d_model, nhead=nhead,
                                                              dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformerlayer_fine = TransformerStyleMixing(d_model=d_model, nhead=nhead,
                                                            dim_feedforward=dim_feedforward, dropout=dropout)
        self.mode = mode
        self.load_weights(name)
        self.iter = 1

    def load_weights(self, name=None):
        if self.mode == 'train':
            ckpt = torch.load(
                ('/home/project/바탕화면/Capstone_Design/JoJoGAN/style_transformer/checkpoints/style_transformer_ffhq.pt'),
                map_location='cpu')
        elif self.mode == 'test':
            ckpt = torch.load(
                (f'/home/project/바탕화면/Capstone_Design/JoJoGAN/models/tf_middle_{name}.pt'),
                map_location='cpu')

        elif self.mode == 'flask':
            ckpt = torch.load((f'demo_flask/static/models/tf/{name}.pt'), map_location='cpu')

        self.transformerlayer_coarse.load_state_dict(get_keys(ckpt, 'transformerlayer_coarse'),
                                                     strict=False)
        self.transformerlayer_medium.load_state_dict(get_keys(ckpt, 'transformerlayer_medium'),
                                                     strict=False)
        self.transformerlayer_fine.load_state_dict(get_keys(ckpt, 'transformerlayer_fine'), strict=False)

    def forward(self, latent, random):
        # coarse = self.transformerlayer_coarse(latent[:4], random[:4])
        # medium = self.transformerlayer_medium(latent[4:8], random[4:8], iter=self.iter, step='middle')
        fine = self.transformerlayer_fine(latent[8:], random[8:], iter=self.iter, step='fine')

        # # attn_style = torch.cat([coarse, medium, fine], dim=0)
        # attn_style = torch.cat([latent[:4], medium, fine], dim=0)
        attn_style = torch.cat([latent[:4], latent[4:8], fine], dim=0)

        self.iter += 1
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
