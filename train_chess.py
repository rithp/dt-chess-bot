"""
train_chess.py
Main training script for the Chess Decision Transformer.

Step 1 — Parse PGN data (if not already done):
    python data/parse_pgn.py \\
        --pgn_dir "/Users/rithvikp/Downloads/Lichess Elite Database" \\
        --out data/chess_trajectories.pkl \\
        --max_games 50000 \\
        --min_year 2018

Step 2 — Train (fresh):
    python train_chess.py \\
        --data_path data/chess_trajectories.pkl \\
        --K 20 --embed_dim 128 --n_layer 3 --n_head 4 \\
        --batch_size 64 --max_iters 10 --num_steps_per_iter 5000 \\
        --device mps

Step 3 — Resume from where you left off:
    python train_chess.py \\
        --data_path data/chess_trajectories.pkl \\
        --K 20 --embed_dim 128 --n_layer 3 --n_head 4 \\
        --batch_size 64 --max_iters 20 --num_steps_per_iter 5000 \\
        --device mps --resume
"""

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

# ── Project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.dataset_utils import MOVE_VOCAB_SIZE, PADDING_IDX
from chess_dt.models.chess_decision_transformer import ChessDecisionTransformer
from chess_dt.training.chess_seq_trainer import ChessSequenceTrainer
from chess_dt.evaluation.evaluate_chess import make_eval_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    """Compute the discounted cumulative sum of a reward array."""
    out = np.zeros_like(x)
    out[-1] = x[-1]
    for t in reversed(range(len(x) - 1)):
        out[t] = x[t] + gamma * out[t + 1]
    return out


def build_get_batch(trajectories, K, batch_size_default, device):
    """
    Returns a get_batch() closure compatible with ChessSequenceTrainer.

    Each trajectory has:
        observations : (T, 768)  float32
        actions      : (T,)      int32
        rewards      : (T,)      float32
        terminals    : (T,)      bool
    """
    # Pre-compute returns (undiscounted sum = game outcome repeated)
    returns = np.array([t["rewards"].sum() for t in trajectories])
    traj_lens = np.array([len(t["actions"]) for t in trajectories])
    num_trajectories = len(trajectories)

    # Sample proportional to game length (more positions = more training signal)
    p_sample = traj_lens / traj_lens.sum()

    print(f"Dataset: {num_trajectories} games  |  "
          f"avg length {traj_lens.mean():.1f}  |  "
          f"total positions {traj_lens.sum():,}")
    print(f"Outcomes — wins: {(returns > 0).sum()}, "
          f"draws: {(returns == 0).sum()}, "
          f"losses: {(returns < 0).sum()}")

    # Global state statistics for (optional) normalization
    # For bitboards this is not strictly needed but harmless
    all_obs = np.concatenate([t["observations"] for t in trajectories], axis=0)
    state_mean = all_obs.mean(axis=0).astype(np.float32)
    state_std  = (all_obs.std(axis=0) + 1e-6).astype(np.float32)
    del all_obs  # free memory

    def get_batch(batch_size=batch_size_default, max_len=K):
        # Sample trajectory indices proportional to length
        batch_inds = np.random.choice(
            num_trajectories, size=batch_size, replace=True, p=p_sample
        )

        s, a, r, d, rtg, ts, mask = [], [], [], [], [], [], []

        for idx in batch_inds:
            traj = trajectories[idx]
            T    = len(traj["actions"])
            # Random start position within the trajectory
            si = random.randint(0, max(0, T - 1))

            # Slice up to max_len steps
            obs_slice = traj["observations"][si: si + max_len]   # (<= K, 768)
            act_slice = traj["actions"][si: si + max_len]        # (<= K,)
            rew_slice = traj["rewards"][si: si + max_len]        # (<= K,)
            done_slice = traj["terminals"][si: si + max_len]     # (<= K,)
            tlen = len(act_slice)

            # RTG at each step = undiscounted sum from step onwards
            # For sparse outcome games this is simply the outcome value
            rtg_full = discount_cumsum(traj["rewards"][si:], gamma=1.0)
            rtg_slice = rtg_full[: tlen + 1].reshape(-1, 1)  # (tlen+1, 1)
            if rtg_slice.shape[0] <= tlen:
                rtg_slice = np.concatenate([rtg_slice, np.zeros((1, 1))], axis=0)

            # Timestep indices
            ts_slice = np.arange(si, si + tlen)

            # ── Padding (left-pad to max_len) ─────────────────────────────
            pad = max_len - tlen

            # States: pad with zeros, then (optional) normalize
            obs_padded = np.concatenate(
                [np.zeros((pad, 768), dtype=np.float32), obs_slice], axis=0
            )
            # Normalize bitboard states
            obs_padded = (obs_padded - state_mean) / state_std

            # Actions: pad with PADDING_IDX
            act_padded = np.concatenate(
                [np.full(pad, PADDING_IDX, dtype=np.int32), act_slice]
            )

            # Rewards
            rew_padded = np.concatenate(
                [np.zeros(pad, dtype=np.float32), rew_slice]
            )

            # Dones
            done_padded = np.concatenate(
                [np.ones(pad, dtype=bool), done_slice]
            )

            # RTG
            rtg_padded = np.concatenate(
                [np.zeros((pad, 1), dtype=np.float32), rtg_slice[: tlen + 1]], axis=0
            )  # shape (max_len+1, 1)

            # Timesteps: pad with 0
            ts_padded = np.concatenate([np.zeros(pad, dtype=np.int64), ts_slice])

            # Attention mask: 0 for padding, 1 for real
            mask_padded = np.concatenate([np.zeros(pad), np.ones(tlen)])

            s.append(obs_padded)
            a.append(act_padded)
            r.append(rew_padded)
            d.append(done_padded)
            rtg.append(rtg_padded)
            ts.append(ts_padded)
            mask.append(mask_padded)

        # ── Tensorise ────────────────────────────────────────────────────────
        def t32(arr):  return torch.tensor(np.array(arr), dtype=torch.float32, device=device)
        def t64(arr):  return torch.tensor(np.array(arr), dtype=torch.long,    device=device)
        def tbool(arr): return torch.tensor(np.array(arr), dtype=torch.float32, device=device)

        return (
            t32(s),        # (B, K, 768)
            t64(a),        # (B, K)
            t32(r).unsqueeze(-1),   # (B, K, 1)
            tbool(d),      # (B, K)
            t32(rtg),      # (B, K+1, 1)
            t64(ts),       # (B, K)
            tbool(mask),   # (B, K)
        )

    return get_batch, state_mean, state_std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Chess Decision Transformer")

    # Data
    parser.add_argument("--data_path",  type=str, default="data/chess_trajectories.pkl",
                        help="Path to pre-parsed trajectory pickle file")
    parser.add_argument("--pct_traj",   type=float, default=1.0,
                        help="Top fraction of trajectories to train on (by return)")

    # Model
    parser.add_argument("--K",          type=int,   default=20,     help="Context window length")
    parser.add_argument("--embed_dim",  type=int,   default=128,    help="Hidden size")
    parser.add_argument("--n_layer",    type=int,   default=3,      help="Number of transformer layers")
    parser.add_argument("--n_head",     type=int,   default=4,      help="Number of attention heads")
    parser.add_argument("--dropout",    type=float, default=0.1,    help="Dropout rate")

    # Training
    parser.add_argument("--batch_size",         type=int,   default=64)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay",  "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps",  type=int,   default=10_000)
    parser.add_argument("--max_iters",     type=int,   default=10)
    parser.add_argument("--num_steps_per_iter", type=int, default=5_000)

    # Evaluation
    parser.add_argument("--num_eval_games", type=int, default=20)
    parser.add_argument("--target_rtg",     type=float, default=1.0,
                        help="RTG to condition on during evaluation (1.0 = aim to win)")

    # Misc
    parser.add_argument("--device",       type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path",    type=str, default="checkpoints/chess_dt_best.pt",
                        help="Path for the best-win-rate checkpoint")
    parser.add_argument("--latest_path",  type=str, default="checkpoints/chess_dt_latest.pt",
                        help="Path for the latest checkpoint (saved every iteration)")
    parser.add_argument("--resume",       action="store_true",
                        help="Resume training from --latest_path checkpoint")
    parser.add_argument("--log_to_wandb", action="store_true")

    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading dataset from {args.data_path} ...")
    with open(args.data_path, "rb") as f:
        trajectories = pickle.load(f)

    # Optionally restrict to top-pct_traj by outcome
    if args.pct_traj < 1.0:
        returns = np.array([t["rewards"].sum() for t in trajectories])
        threshold = np.quantile(returns, 1.0 - args.pct_traj)
        trajectories = [t for t, r in zip(trajectories, returns) if r >= threshold]
        print(f"Using top {args.pct_traj*100:.0f}% of trajectories → {len(trajectories)} games")

    get_batch, state_mean, state_std = build_get_batch(
        trajectories, args.K, args.batch_size, args.device
    )

    # ── Build model ──────────────────────────────────────────────────────────
    model = ChessDecisionTransformer(
        state_dim        = 768,
        move_vocab_size  = MOVE_VOCAB_SIZE,
        padding_idx      = PADDING_IDX,
        hidden_size      = args.embed_dim,
        max_length       = args.K,
        max_ep_len       = 512,
        n_layer          = args.n_layer,
        n_head           = args.n_head,
        n_inner          = 4 * args.embed_dim,
        activation_function = "relu",
        resid_pdrop      = args.dropout,
        attn_pdrop       = args.dropout,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters")

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / args.warmup_steps, 1.0),
    )

    # ── Evaluation function ──────────────────────────────────────────────────
    eval_fn = make_eval_fn(
        num_games  = args.num_eval_games,
        target_rtg = args.target_rtg,
        device     = args.device,
    )

    # ── Trainer ─────────────────────────────────────────────────────────────
    trainer = ChessSequenceTrainer(
        model      = model,
        optimizer  = optimizer,
        batch_size = args.batch_size,
        get_batch  = get_batch,
        scheduler  = scheduler,
        eval_fns   = [eval_fn],
    )

    # ── WandB (optional) ────────────────────────────────────────────────────
    if args.log_to_wandb:
        import wandb
        wandb.init(
            project="chess-decision-transformer",
            config=vars(args),
            name=f"chess-dt-K{args.K}-d{args.embed_dim}-L{args.n_layer}",
        )

    # ── Training loop ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.save_path)   or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.latest_path) or ".", exist_ok=True)

    start_iter    = 0
    best_win_rate = -1.0

    # ── Resume from latest checkpoint (if requested) ─────────────────────────
    if args.resume:
        resume_path = args.latest_path
        if not os.path.exists(resume_path):
            print(f"[resume] No checkpoint found at {resume_path}, starting fresh.")
        else:
            print(f"[resume] Loading checkpoint from {resume_path} ...")
            ckpt = torch.load(resume_path, map_location=args.device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_iter    = ckpt.get("iteration", 0) + 1
            best_win_rate = ckpt.get("best_win_rate", -1.0)
            print(f"[resume] Resuming from iteration {start_iter + 1}  "
                  f"(best win rate so far: {best_win_rate:.3f})")
    for iteration in range(start_iter, args.max_iters):
        outputs = trainer.train_iteration(
            num_steps  = args.num_steps_per_iter,
            iter_num   = iteration + 1,
            print_logs = True,
        )

        if args.log_to_wandb:
            import wandb
            wandb.log(outputs)

        win_key  = f"eval/win_rate_rtg{args.target_rtg:.1f}"
        win_rate = outputs.get(win_key, 0.0)

        # ── Always save the latest checkpoint ────────────────────────────────
        latest_payload = {
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "iteration":            iteration,
            "best_win_rate":        best_win_rate,
            "args":                 vars(args),
            "state_mean":           state_mean,
            "state_std":            state_std,
        }
        torch.save(latest_payload, args.latest_path)
        print(f"  → Latest checkpoint saved (iter {iteration + 1})")

        # ── Save best checkpoint only on improvement ──────────────────────────
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            torch.save(latest_payload | {"best_win_rate": best_win_rate}, args.save_path)
            print(f"  ✓ Best checkpoint saved (win rate {best_win_rate:.3f})")

    print("\nTraining complete.")
    print(f"Best win rate vs random: {best_win_rate:.3f}")



if __name__ == "__main__":
    main()
