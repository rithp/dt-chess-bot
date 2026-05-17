"""
trajectory_gpt2.py
Self-contained GPT-2 backbone adapted from kzl/decision-transformer.

Key changes from the original:
  - Positional embeddings are removed; DT injects its own timestep embeddings.
  - GPT2Model is a plain nn.Module (not PreTrainedModel) for transformers 5.x compat.
  - Uses transformers GPT2Config for hyperparameters, ACT2FN for activations.
"""

import torch
import torch.nn as nn

from transformers.activations import ACT2FN
from transformers.pytorch_utils import Conv1D
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, nx: int, n_ctx: int, config: GPT2Config, scale: bool = False):
        super().__init__()
        n_state = nx
        assert n_state % config.n_head == 0, \
            f"n_embd ({n_state}) must be divisible by n_head ({config.n_head})"

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(n_ctx, n_ctx, dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head    = config.n_head
        self.split_size = n_state
        self.scale     = scale

        self.c_attn    = Conv1D(3 * n_state, nx)
        self.c_proj    = Conv1D(n_state, nx)
        self.attn_dropout  = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)
        nd, ns = w.size(-2), w.size(-1)
        mask = self.bias[:, :, ns - nd: ns, :ns]
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))
        if attention_mask is not None:
            w = w + attention_mask
        w = torch.softmax(w, dim=-1)
        w = self.attn_dropout(w)
        if head_mask is not None:
            w = w * head_mask
        return torch.matmul(w, v)

    def _split_heads(self, x, k: bool = False):
        new_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_shape)
        return x.permute(0, 2, 3, 1) if k else x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_shape)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache: bool = False,
    ):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query)
        key   = self._split_heads(key, k=True)
        value = self._split_heads(value)

        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
            key   = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        present = torch.stack((key.transpose(-2, -1), value)) if use_cache else (None,)

        a = self._attn(query, key, value, attention_mask, head_mask)
        a = self._merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a, present


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, n_state: int, config: GPT2Config):
        super().__init__()
        nx = config.n_embd
        self.c_fc    = Conv1D(n_state, nx)
        self.c_proj  = Conv1D(nx, n_state)
        self.act     = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, n_ctx: int, config: GPT2Config, scale: bool = False):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim   = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1  = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn  = Attention(hidden_size, n_ctx, config, scale)
        self.ln_2  = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp   = MLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache: bool = False,
    ):
        attn_out, present = self.attn(
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states, present


# ---------------------------------------------------------------------------
# GPT-2 Model (plain nn.Module — no HF PreTrainedModel dependency)
# ---------------------------------------------------------------------------

class GPT2Model(nn.Module):
    """
    Lightweight GPT-2 backbone.

    Differences from HuggingFace GPT2Model:
      - No positional embeddings (DT provides timestep embeddings)
      - Plain nn.Module — no PreTrainedModel API surface
      - forward() always returns a dict with 'last_hidden_state'
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # Context length: GPT2Config used 'n_ctx' historically; v5 uses 'n_positions'
        n_ctx = getattr(config, "n_ctx", None) or config.n_positions

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h    = nn.ModuleList([Block(n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, Conv1D):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(
        self,
        inputs_embeds,          # (B, seq_len, n_embd)
        attention_mask=None,    # (B, seq_len) — 1 attend / 0 ignore
        use_cache: bool = False,
    ) -> dict:
        B, T, _ = inputs_embeds.shape
        hidden_states = self.drop(inputs_embeds)

        # Convert attention mask to additive format
        if attention_mask is not None:
            # (B, 1, 1, T)  — broadcast over heads and query positions
            attn_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attn_mask = (1.0 - attn_mask) * -10000.0
        else:
            attn_mask = None

        presents    = [] if use_cache else None
        past_kvs    = [None] * len(self.h)

        for i, (block, past) in enumerate(zip(self.h, past_kvs)):
            hidden_states, present = block(
                hidden_states,
                layer_past     = past,
                attention_mask = attn_mask,
                use_cache      = use_cache,
            )
            if use_cache:
                presents.append(present)

        hidden_states = self.ln_f(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values":   presents,
        }
