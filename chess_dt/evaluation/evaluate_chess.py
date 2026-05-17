"""
evaluate_chess.py
Evaluation loop for the Chess Decision Transformer.

Plays complete games against a baseline opponent (random mover by default)
and tracks win / draw / loss statistics.

Usage (from train_chess.py):
    from chess_dt.evaluation.evaluate_chess import make_eval_fn
    eval_fn = make_eval_fn(num_games=20, target_rtg=1.0, device="cuda")
    results = eval_fn(model)
"""

import chess
import numpy as np
import torch

from data.dataset_utils import (
    board_to_state,
    move_to_index,
    index_to_move,
    legal_move_mask,
    MOVE_VOCAB_SIZE,
    PADDING_IDX,
)


# ---------------------------------------------------------------------------
# Opponents
# ---------------------------------------------------------------------------

def random_opponent(board: chess.Board) -> chess.Move:
    """Pick a uniformly random legal move."""
    moves = list(board.legal_moves)
    return np.random.choice(moves)


# ---------------------------------------------------------------------------
# Single-game rollout
# ---------------------------------------------------------------------------

def play_game(
    model,
    bot_color: chess.Color,
    target_rtg: float = 1.0,
    max_moves: int = 400,
    device: str = "cpu",
    opponent_fn=None,
) -> str:
    """
    Play a single game.  The DT controls `bot_color`; the opponent plays the
    other side with `opponent_fn` (default: random).

    Returns the PGN result string: '1-0', '0-1', or '1/2-1/2'.
    """
    if opponent_fn is None:
        opponent_fn = random_opponent

    board = chess.Board()

    # History buffers (grow as the game progresses)
    states_buf    = []
    actions_buf   = []
    rtg_buf       = []
    timesteps_buf = []

    current_rtg = target_rtg

    for move_num in range(max_moves):
        if board.is_game_over():
            break

        if board.turn == bot_color:
            # ── DT's turn ────────────────────────────────────────────────────
            state = board_to_state(board)
            states_buf.append(state)
            rtg_buf.append(current_rtg)
            timesteps_buf.append(move_num)

            # Build tensors
            T = len(states_buf)
            states_t    = torch.tensor(np.array(states_buf),    dtype=torch.float32, device=device)
            rtg_t       = torch.tensor(np.array(rtg_buf),       dtype=torch.float32, device=device).unsqueeze(-1)
            timesteps_t = torch.tensor(np.array(timesteps_buf), dtype=torch.long,    device=device)

            # Pad actions buffer to same length (last action unknown → PADDING_IDX)
            if len(actions_buf) < T:
                padded_actions = actions_buf + [PADDING_IDX]
            else:
                padded_actions = actions_buf[-T:]
            actions_t = torch.tensor(padded_actions, dtype=torch.long, device=device)

            lmask = legal_move_mask(board)

            move_idx = model.get_action(
                states    = states_t,
                actions   = actions_t,
                returns_to_go = rtg_t,
                timesteps = timesteps_t,
                legal_mask = lmask,
                device    = device,
            )
            move = index_to_move(move_idx, board)

            # Validate — fall back to random if decoding produced illegal move
            if move not in board.legal_moves:
                move = random_opponent(board)
                move_idx = move_to_index(move)

            actions_buf.append(move_idx)

        else:
            # ── Opponent's turn ──────────────────────────────────────────────
            move = opponent_fn(board)

        board.push(move)

    result = board.result(claim_draw=True)
    return result


# ---------------------------------------------------------------------------
# Evaluation function factory (compatible with ChessSequenceTrainer)
# ---------------------------------------------------------------------------

def make_eval_fn(
    num_games: int = 20,
    target_rtg: float = 1.0,
    device: str = "cpu",
    opponent_fn=None,
    bot_color: chess.Color = chess.WHITE,
):
    """
    Returns a function  eval_fn(model) → dict  that plays `num_games` games
    and reports win/draw/loss rates plus move accuracy.
    """
    def eval_fn(model) -> dict:
        model.eval()
        results = []
        with torch.no_grad():
            for _ in range(num_games):
                result = play_game(
                    model      = model,
                    bot_color  = bot_color,
                    target_rtg = target_rtg,
                    device     = device,
                    opponent_fn = opponent_fn,
                )
                results.append(result)

        wins   = sum(1 for r in results if (r == "1-0" and bot_color == chess.WHITE)
                                        or (r == "0-1" and bot_color == chess.BLACK))
        draws  = sum(1 for r in results if r == "1/2-1/2")
        losses = num_games - wins - draws

        return {
            f"eval/win_rate_rtg{target_rtg:.1f}":   wins   / num_games,
            f"eval/draw_rate_rtg{target_rtg:.1f}":  draws  / num_games,
            f"eval/loss_rate_rtg{target_rtg:.1f}":  losses / num_games,
        }

    return eval_fn
