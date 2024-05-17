"""Implementation of the ProteinBERT model.

This code has been modified from the original implementation
by Facebook Research, describing its ESM-1b paper."""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from typing_extensions import override
import torch
from torch import nn

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
        default=12, metadata=dict(help="number of transformer layers")
    )
    embed_dim: int = field(default=768, metadata=dict(help="embedding dimension"))
    attention_dropout: float = field(
        default=0.0, metadata=dict(help="dropout on attention")
    )
    logit_bias: bool = field(
        default=False,
        metadata=dict(help="whether to apply bias to logits"),
    )
    no_rope_embedding: bool = field(
        default=False,
        metadata=dict(help="don't use Rotary Positional Embeddings"),
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
    token_dropout: bool = False
    alphabet: str = "CodonModel"

    @override
    def to_namespace(self, max_positions: int = 1024, **_kwargs):
        return super().to_namespace(max_positions=max_positions)


class ProteinBertModel(nn.Module):
    layers: nn.Module
    embed_tokens: nn.Module
    emb_layer_norm_after: nn.Module
    lm_head: nn.Module

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        return ProteinBertModelCfg.add_args(parser)

    @classmethod
    def create_cfg(cls, args: Namespace) -> ProteinBertModelCfg:
        return ProteinBertModelCfg.create(args)

    def __init__(self, args: Namespace):
        super().__init__()
        self.cfg = ProteinBertModel.create_cfg(args)
        max_positions = getattr(args, "max_positions", None)
        if max_positions is None:
            raise ValueError("no max_positions in args Namespace")
        self.max_positions = int(max_positions)

        self.alphabet = alphabet = Alphabet.from_architecture(self.cfg.alphabet)
        self.alphabet_size = len(alphabet)

        self.model_version = "ESM-1b"
        self.embed_scale = 1
        self.emb_layer_norm_before: nn.Module | None = None
        self.embed_positions: nn.Module | None = None

        self._init_submodules_esm1b()

    def _init_submodules_common(self):
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,  # 69 for RNA
            self.cfg.embed_dim,  # 768
            padding_idx=self.alphabet.padding_idx,
        )
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.cfg.embed_dim,
                    self.cfg.ffn_embed_dim,  # 4*768
                    self.cfg.attention_heads,  # 12
                    attention_dropout=self.cfg.attention_dropout,
                    add_bias_kv=(self.model_version != "ESM-1b"),
                    use_esm1b_layer_norm=(self.model_version == "ESM-1b"),
                    rope_embedding=not self.cfg.no_rope_embedding,
                )
                for _ in range(self.cfg.num_layers)
            ]
        )

    def _init_submodules_esm1b(self):
        self._init_submodules_common()
        if self.cfg.no_rope_embedding:
            self.embed_positions = LearnedPositionalEmbedding(
                self.max_positions, self.cfg.embed_dim, self.alphabet.padding_idx
            )
        if self.cfg.emb_layer_norm_before:
            self.emb_layer_norm_before = ESM1bLayerNorm(self.cfg.embed_dim)

        self.emb_layer_norm_after = ESM1bLayerNorm(self.cfg.embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=self.cfg.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        repr_layers: list[int] | set[int] | None = None,
        need_head_weights: bool = False,
    ):
        padding_mask: torch.Tensor | None
        if repr_layers is None:
            repr_layers = []

        assert tokens.ndim == 2  # batch_size x  max_positions
        padding_mask = tokens.eq(self.alphabet.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.cfg.token_dropout:
            x.masked_fill_((tokens == self.alphabet.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            x.masked_fill_((tokens == self.alphabet.mask_idx).unsqueeze(-1), 0.0)
            mask_ratio_observed = (tokens == self.alphabet.mask_idx).sum(
                -1
            ).float() / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if self.embed_positions is not None:
            x = x + self.embed_positions(tokens)

        if self.emb_layer_norm_before is not None:
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
        layer_idx = 0
        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if layer_idx + 1 in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if layer_idx + 1 in repr_layers:
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

        return result  # logits = batch x max_positions x alphabet_size

    @property
    def num_layers(self) -> int:
        return self.cfg.num_layers
