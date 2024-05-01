"""Implementation of the ProteinBERT model.

This code has been modified from the original implementation
by Facebook Research, describing its ESM-1b paper."""

from argparse import ArgumentParser, Namespace
import math
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    TransformerLayer,
    LearnedPositionalEmbedding,
    RobertaLMHead,
    ESM1bLayerNorm,
)

from .alphabet import Alphabet
from .utils import ArgparseMixin


@dataclass
class ProteinBertModelCfg(ArgparseMixin):
    num_layers: int = field(
        default=12, metadata=dict(help="number of layers")
    )
    embed_dim: int = field(
        default=768, metadata=dict(help="embedding dimension")
    )
    attention_dropout: float = field(
        default=0.0, metadata=dict(help="dropout on attention")
    )
    logit_bias: bool = field(
        default=False,
        metadata=dict(help="whether to apply bias to logits"),
    )
    rope_embedding: bool = field(
        default=True,
        metadata=dict(help="whether to use Rotary Positional Embeddings"),
    )
    ffn_embed_dim: int = field(
        default=768 * 4,
        metadata=dict(help="embedding dimension for FFN"),
    )
    attention_heads: int = field(
        default=12,
        metadata=dict(help="number of attention heads"),
    )
    emb_layer_norm_before: bool = False


class ProteinBertModel(nn.Module):

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        return ProteinBertModelCfg.add_args(parser)

    @classmethod
    def create(cls, args: Namespace) -> ProteinBertModelCfg:
        return ProteinBertModelCfg.create(args)

    def __init__(self, args: ProteinBertModelCfg, alphabet: Alphabet):
        super().__init__()
        self.args = args
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.emb_layer_norm_before = self.args.emb_layer_norm_before
        self.model_version = "ESM-1b"
        self._init_submodules_esm1b()

    def _init_submodules_common(self):
        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.args.embed_dim, padding_idx=self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.args.embed_dim,
                    self.args.ffn_embed_dim,
                    self.args.attention_heads,
                    attention_dropout=self.args.attention_dropout,
                    add_bias_kv=(self.model_version != "ESM-1b"),
                    use_esm1b_layer_norm=(self.model_version == "ESM-1b"),
                    rope_embedding=self.args.rope_embedding,
                )
                for _ in range(self.args.num_layers)
            ]
        )

    def _init_submodules_esm1b(self):
        self._init_submodules_common()
        self.embed_scale = 1
        if not self.args.rope_embedding:
            self.embed_positions = LearnedPositionalEmbedding(
                self.args.max_positions, self.args.embed_dim, self.padding_idx
            )
        self.emb_layer_norm_before = (
            ESM1bLayerNorm(self.args.embed_dim) if self.emb_layer_norm_before else None
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=self.args.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def forward(self, tokens, repr_layers=[], need_head_weights=False):

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if getattr(self.args, "token_dropout", False):
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(
                -1
            ).float() / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if not self.args.rope_embedding:
            x = x + self.embed_positions(tokens)

        if self.emb_layer_norm_before:
            x = self.emb_layer_norm_before(x)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if self.model_version == "ESM-1":
                # ESM-1 models have an additional null-token for attention, which we remove
                attentions = attentions[..., :-1]
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(
                    2
                )
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions

        return result

    @property
    def num_layers(self):
        return self.args.num_layers
