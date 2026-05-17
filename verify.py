"""
verify.py — Quick sanity check for the Chess Decision Transformer.
Run from the project root: python verify.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import chess

print("=" * 60)
print("STEP 1: Board/move encoding")
from data.dataset_utils import (
    board_to_state, move_to_index, index_to_move,
    legal_move_mask, MOVE_VOCAB_SIZE, PADDING_IDX
)

board = chess.Board()
state = board_to_state(board)
assert state.shape == (768,), f"Bad state shape: {state.shape}"
assert state.dtype == np.float32
print(f"  board_to_state → shape {state.shape}, sum={state.sum():.0f}  ✓")

# e2e4 = from e2(12) to e4(28)
move = chess.Move.from_uci("e2e4")
idx  = move_to_index(move)
assert 0 <= idx < MOVE_VOCAB_SIZE, f"Bad index: {idx}"
print(f"  move e2e4 → index {idx}  ✓")

decoded = index_to_move(idx, board)
assert decoded in board.legal_moves, f"Decoded move {decoded} illegal"
print(f"  index {idx} → move {decoded.uci()}  ✓")

lmask = legal_move_mask(board)
assert lmask.shape == (MOVE_VOCAB_SIZE,)
assert lmask.sum() == len(list(board.legal_moves))
print(f"  legal_move_mask → {lmask.sum()} legal moves  ✓")

print()
print("STEP 2: Model forward pass")
from chess_dt.models.chess_decision_transformer import ChessDecisionTransformer

model = ChessDecisionTransformer(
    hidden_size=64,
    max_length=4,
    max_ep_len=64,
    n_layer=2,
    n_head=2,
)
model.eval()

B, K = 2, 4
states        = torch.zeros(B, K, 768)
actions       = torch.full((B, K), PADDING_IDX, dtype=torch.long)
returns_to_go = torch.ones(B, K, 1)
timesteps     = torch.arange(K).unsqueeze(0).expand(B, -1)
attn_mask     = torch.ones(B, K)

with torch.no_grad():
    logits = model(states, actions, returns_to_go, timesteps, attn_mask)

assert logits.shape == (B, K, MOVE_VOCAB_SIZE), f"Bad logit shape: {logits.shape}"
assert not torch.isnan(logits).any(), "NaN in logits!"
print(f"  Forward pass → logits shape {tuple(logits.shape)}  ✓")

print()
print("STEP 3: get_action (greedy move selection)")
board2 = chess.Board()
lmask  = legal_move_mask(board2)
s_hist = torch.tensor(board_to_state(board2)).unsqueeze(0)  # (1, 768)
a_hist = torch.full((1,), PADDING_IDX, dtype=torch.long)
r_hist = torch.ones(1, 1)  # RTG = 1
t_hist = torch.zeros(1, dtype=torch.long)

with torch.no_grad():
    chosen_idx = model.get_action(s_hist, a_hist, r_hist, t_hist, legal_mask=lmask)

assert 0 <= chosen_idx < MOVE_VOCAB_SIZE
move = index_to_move(chosen_idx, board2)
print(f"  get_action → index {chosen_idx} → move {move.uci()}  ✓")

print()
print("STEP 4: Loss computation")
import torch.nn.functional as F
logits_flat  = logits.reshape(B * K, MOVE_VOCAB_SIZE)
targets      = torch.zeros(B * K, dtype=torch.long)
loss = F.cross_entropy(logits_flat, targets)
assert not torch.isnan(loss), "NaN loss!"
print(f"  Cross-entropy loss = {loss.item():.4f}  ✓")

print()
print("=" * 60)
print("All checks passed! ✓")
print(f"MOVE_VOCAB_SIZE = {MOVE_VOCAB_SIZE}  |  PADDING_IDX = {PADDING_IDX}")
n_params = sum(p.numel() for p in model.parameters())
print(f"Mini model params: {n_params:,}")
