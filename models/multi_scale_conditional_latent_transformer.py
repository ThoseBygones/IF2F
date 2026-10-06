"""Multi-scale conditional latent transformer used by IF2F.

The public class names intentionally match the classes used to train the
archived checkpoints. Renaming them would not change ``state_dict`` keys, but
keeping the names makes the relationship with the original implementation
explicit and avoids unnecessary compatibility risk.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TransformerDecoderLayer(nn.Module):
    """Self-attention and hash-conditioned cross-attention block."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        normalize_type="layernorm",
    ):
        super().__init__()
        assert normalize_type in ["layernorm", "instancenorm", "None"]
        self.normalize_type = normalize_type
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if normalize_type == "layernorm":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        elif normalize_type == "instancenorm":
            self.norm1 = nn.InstanceNorm1d(d_model, affine=True)
            self.norm2 = nn.InstanceNorm1d(d_model, affine=True)
            self.norm3 = nn.InstanceNorm1d(d_model, affine=True)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        if self.normalize_type != "None":
            tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        if self.normalize_type != "None":
            tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        if self.normalize_type != "None":
            tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        return tgt + self.dropout3(tgt2)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


class TransformerMapperSplitMulti(nn.Module):
    """The MCLT implementation used by the final IF2F checkpoint."""

    def __init__(
        self,
        block_num=1,
        split_list=[4, 4, 6],
        normalize_type="layernorm",
        add_linear=False,
        add_pos_embedding=False,
    ):
        super(TransformerMapperSplitMulti, self).__init__()
        assert normalize_type in ["layernorm", "instancenorm", "None"]
        self.layers = nn.ModuleList()
        self.block_num = block_num
        self.split_list = split_list
        self.add_linear = add_linear
        self.add_pos_embedding = add_pos_embedding

        if self.add_linear:
            self.linear_ori = nn.Linear(512, 512)
            self.linear_pwd = nn.Linear(512, 512)
        if self.add_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.randn(sum(self.split_list), 1, 512))

        for _ in range(block_num):
            layer_list = []
            for _ in range(len(split_list)):
                layer_list.append(
                    TransformerDecoderLayer(
                        d_model=512,
                        nhead=4,
                        dim_feedforward=1024,
                        normalize_type=normalize_type,
                    )
                )
            self.layers.append(nn.ModuleList(layer_list))

    def split_latent(self, latent):
        return torch.split(latent, self.split_list, dim=0)

    def forward(self, input_tensor):
        w_ori, w_pwd = torch.chunk(input_tensor, 2, dim=-1)
        w_ori = w_ori.permute(1, 0, 2)
        w_pwd = w_pwd.permute(1, 0, 2)

        if self.add_linear:
            w_ori = self.linear_ori(w_ori)
            w_pwd = self.linear_pwd(w_pwd)
        if self.add_pos_embedding:
            w_pos_split = self.split_latent(self.pos_embedding)

        w_ori_split = self.split_latent(w_ori)
        w_pwd_split = self.split_latent(w_pwd)

        w_out_list = []
        for index in range(len(self.layers[0])):
            if self.add_pos_embedding:
                w_split_temp = w_ori_split[index] + w_pos_split[index]
            else:
                w_split_temp = w_ori_split[index]
            w_out_temp = self.layers[0][index](w_split_temp, w_pwd_split[index])
            w_out_list.append(w_out_temp)
        w_out = torch.cat(w_out_list, dim=0)

        # Preserve the original iteration behavior for checkpoint compatibility.
        for block_index in range(self.block_num - 1):
            w_out_split = self.split_latent(w_out)
            w_out_list = []
            for index in range(len(self.layers[block_index])):
                w_out_temp = self.layers[block_index][index](
                    w_out_split[index], w_pwd_split[index]
                )
                w_out_list.append(w_out_temp)
            w_out = torch.cat(w_out_list, dim=0)

        return w_out.permute(1, 0, 2)
