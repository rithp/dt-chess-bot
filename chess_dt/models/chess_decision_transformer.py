"""
chess_decision_transformer.py
Chess-adapted Decision Transformer.

Key differences from the original kzl/DT:
  - State:  768-dim bitboard float vector (via Linear projection)
  - Action: discrete integer index in [0, MOVE_VOCAB_SIZE) (via Embedding)
  - Head:   Linear(hidden_size, MOVE_VOCAB_SIZE) — cross-entropy loss, not MSE
  - RTG:    sparse game outcome in {-1, 0, +1}, fed as a scalar (via Linear)
"""

import torch
import torch.nn as nn
import transformers

from chess_dt.models.trajectory_gpt2 import GPT2Model
from data.dataset_utils import MOVE_VOCAB_SIZE, PADDING_IDX


class ChessDecisionTransformer(nn.Module):
    """
    Decision Transformer for chess.

    Sequence structure (per timestep t):
        [RTG_t, state_t, action_t]  →  stacked into 3*K tokens for GPT-2

    The model predicts the next action distribution from the state token output:
        action_logits = predict_action(x[:, 1, t])   # state slot → action

    RTG (return-to-go) is in {-1.0, 0.0, 1.0} — no normalization needed.
    """

    def __init__(
        self,
        state_dim: int = 768,
        move_vocab_size: int = MOVE_VOCAB_SIZE,
        padding_idx: int = PADDING_IDX,
        hidden_size: int = 128,
        max_length: int = 20,        # context window K
        max_ep_len: int = 512,       # max game length for timestep embedding
        n_layer: int = 3,
        n_head: int = 1,
        n_inner: int = None,
        activation_function: str = "relu",
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
    ):
        super().__init__()

        self.state_dim      = state_dim
        self.move_vocab_size = move_vocab_size
        self.padding_idx    = padding_idx
        self.hidden_size    = hidden_size
        self.max_length     = max_length

        config = transformers.GPT2Config(
            vocab_size=1,              # not used — we feed embeddings directly
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner or 4 * hidden_size,
            activation_function=activation_function,
            n_positions=1024,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
        )

        self.transformer = GPT2Model(config)

        # ── Input projections ────────────────────────────────────────────────
        # Timestep embedding (added to all three modalities)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        # RTG: scalar → hidden_size
        self.embed_return   = nn.Linear(1, hidden_size)

        # State: 768-dim bitboard → hidden_size
        self.embed_state    = nn.Linear(state_dim, hidden_size)

        # Action: discrete move index → hidden_size
        # padding_idx masks padded timesteps from contributing to gradients
        self.embed_action   = nn.Embedding(
            move_vocab_size + 1,   # +1 for the padding token
            hidden_size,
            padding_idx=padding_idx,
        )

        self.embed_ln = nn.LayerNorm(hidden_size)

        # ── Output heads ─────────────────────────────────────────────────────
        # Predict logits over all possible moves from the state token
        self.predict_action = nn.Linear(hidden_size, move_vocab_size)

        # Optional auxiliary heads (not used in loss during training)
        self.predict_return = nn.Linear(hidden_size, 1)

    # -------------------------------------------------------------------------

    def forward(
        self,
        states,           # (B, K, 768)          float32
        actions,          # (B, K)                int64   — PADDING_IDX for masked steps
        returns_to_go,    # (B, K, 1)             float32
        timesteps,        # (B, K)                int64
        attention_mask=None,  # (B, K)            float32  0/1
    ):
        """
        Forward pass.  Returns action_logits of shape (B, K, MOVE_VOCAB_SIZE).
        """
        B, K = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((B, K), dtype=torch.float32, device=states.device)

        # ── Embed each modality ───────────────────────────────────────────────
        time_emb     = self.embed_timestep(timesteps)            # (B, K, H)
        state_emb    = self.embed_state(states) + time_emb       # (B, K, H)
        returns_emb  = self.embed_return(returns_to_go) + time_emb  # (B, K, H)
        action_emb   = self.embed_action(actions) + time_emb     # (B, K, H)

        # ── Stack into (R_1, s_1, a_1, R_2, s_2, a_2, ...) ──────────────────
        # Shape after stack + permute: (B, 3K, H)
        stacked_inputs = torch.stack(
            (returns_emb, state_emb, action_emb), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * K, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(B, 3 * K)

        # ── GPT-2 forward ─────────────────────────────────────────────────────
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs["last_hidden_state"]  # (B, 3K, H)


        # Reshape back: x[:, i, t] corresponds to token type i at timestep t
        # 0=return, 1=state, 2=action
        x = x.reshape(B, K, 3, self.hidden_size).permute(0, 2, 1, 3)
        # x[:, 0] → return tokens, x[:, 1] → state tokens, x[:, 2] → action tokens

        # Predict next action from the state token (as in the original DT paper)
        action_logits = self.predict_action(x[:, 1])  # (B, K, MOVE_VOCAB_SIZE)

        return action_logits

    # -------------------------------------------------------------------------

    @torch.no_grad()
    def get_action(
        self,
        states,          # (T, 768)   — history of board states
        actions,         # (T,)       — history of move indices
        returns_to_go,   # (T,)       — RTG at each step
        timesteps,       # (T,)       — timestep indices
        legal_mask=None, # (MOVE_VOCAB_SIZE,) bool — mask illegal moves
        device="cpu",
    ) -> int:
        """
        Greedy action selection for a single game step.
        Pads/truncates to max_length, returns the best legal move index.
        """
        # Add batch dim
        states        = states.reshape(1, -1, self.state_dim)
        actions       = actions.reshape(1, -1)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps     = timesteps.reshape(1, -1)

        K = self.max_length

        # Truncate to context window
        if states.shape[1] > K:
            states        = states[:, -K:]
            actions       = actions[:, -K:]
            returns_to_go = returns_to_go[:, -K:]
            timesteps     = timesteps[:, -K:]

        T = states.shape[1]
        pad = K - T

        # Build attention mask
        attn_mask = torch.cat([
            torch.zeros(1, pad, device=device),
            torch.ones(1, T,   device=device),
        ], dim=1)

        # Pad states with zeros on the left
        if pad > 0:
            states        = torch.cat([torch.zeros(1, pad, self.state_dim, device=device), states], dim=1)
            returns_to_go = torch.cat([torch.zeros(1, pad, 1,             device=device), returns_to_go], dim=1)
            timesteps     = torch.cat([torch.zeros(1, pad,                device=device, dtype=torch.long), timesteps], dim=1)
            # Pad actions with PADDING_IDX
            pad_actions   = torch.full((1, pad), self.padding_idx, dtype=torch.long, device=device)
            actions       = torch.cat([pad_actions, actions], dim=1)

        # Forward
        action_logits = self.forward(
            states.to(dtype=torch.float32),
            actions.to(dtype=torch.long),
            returns_to_go.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            attention_mask=attn_mask,
        )  # (1, K, MOVE_VOCAB_SIZE)

        # Take logits at the last real timestep
        logits = action_logits[0, -1]  # (MOVE_VOCAB_SIZE,)

        # Apply legal move mask
        if legal_mask is not None:
            illegal = torch.tensor(~legal_mask, device=device)
            logits = logits.masked_fill(illegal, float("-inf"))

        return int(torch.argmax(logits).item())
