"""
parse_pgn.py
Converts PGN files from the Lichess Elite Database into trajectory pickle files
compatible with the Chess Decision Transformer.

Each trajectory dict has:
  observations : np.ndarray  shape (T, 768)  float32   bitboard states
  actions      : np.ndarray  shape (T,)      int32     move indices
  rewards      : np.ndarray  shape (T,)      float32   sparse outcome (terminal only)
  terminals    : np.ndarray  shape (T,)      bool      True at the last move

Usage:
  python data/parse_pgn.py \\
      --pgn_dir "/Users/rithvikp/Downloads/Lichess Elite Database" \\
      --out data/chess_trajectories.pkl \\
      --max_games 50000 \\
      --min_year 2018

The script can be re-run with different --max_games to produce subset datasets.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import chess
import chess.pgn
import numpy as np
from tqdm import tqdm

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.dataset_utils import board_to_state, move_to_index, outcome_to_reward, PADDING_IDX


def parse_pgn_file(pgn_path: str, max_games: int, trajectories: list, verbose: bool = True):
    """
    Parse a single PGN file and append trajectories to the provided list.
    Returns the number of games successfully parsed from this file.
    """
    n_parsed = 0
    n_skipped = 0

    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        while len(trajectories) < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            result = game.headers.get("Result", "*")
            if result not in ("1-0", "0-1", "1/2-1/2"):
                n_skipped += 1
                continue

            # Collect moves
            board = game.board()
            observations = []
            actions = []

            for move in game.mainline_moves():
                observations.append(board_to_state(board))
                actions.append(move_to_index(move))
                board.push(move)

            # Skip extremely short games (e.g. illegal/forfeit)
            if len(actions) < 2:
                n_skipped += 1
                continue

            T = len(actions)

            # Sparse reward: only the terminal timestep carries the outcome.
            # We record it from White's perspective always, since the model
            # is trained to predict "what would a strong player do here".
            # During training we always feed the board from the current side-to-move.
            rewards = np.zeros(T, dtype=np.float32)
            rewards[-1] = outcome_to_reward(result, chess.WHITE)

            terminals = np.zeros(T, dtype=bool)
            terminals[-1] = True

            trajectories.append({
                "observations": np.array(observations, dtype=np.float32),
                "actions":      np.array(actions,      dtype=np.int32),
                "rewards":      rewards,
                "terminals":    terminals,
                "result":       result,
            })
            n_parsed += 1

    if verbose:
        print(f"  {os.path.basename(pgn_path)}: {n_parsed} parsed, {n_skipped} skipped")
    return n_parsed


def main():
    parser = argparse.ArgumentParser(description="Parse Lichess Elite PGN files into trajectory pkl")
    parser.add_argument("--pgn_dir",   type=str, required=True,
                        help="Directory containing .pgn files")
    parser.add_argument("--out",       type=str, default="data/chess_trajectories.pkl",
                        help="Output pickle file path")
    parser.add_argument("--max_games", type=int, default=50_000,
                        help="Maximum number of games to parse (default: 50000)")
    parser.add_argument("--min_year",  type=int, default=0,
                        help="Only include PGN files from this year onwards (e.g. 2018)")
    args = parser.parse_args()

    pgn_dir = Path(args.pgn_dir)
    pgn_files = sorted(pgn_dir.glob("*.pgn"))

    if args.min_year > 0:
        pgn_files = [p for p in pgn_files if _year_from_filename(p.name) >= args.min_year]

    if not pgn_files:
        print(f"No PGN files found in {pgn_dir} matching min_year={args.min_year}")
        sys.exit(1)

    print(f"Found {len(pgn_files)} PGN file(s) to process.")
    print(f"Target: {args.max_games} games  |  Output: {args.out}")
    print("=" * 60)

    trajectories = []
    for pgn_path in tqdm(pgn_files, desc="PGN files", unit="file"):
        if len(trajectories) >= args.max_games:
            break
        parse_pgn_file(str(pgn_path), args.max_games, trajectories)

    print("=" * 60)
    print(f"Total trajectories collected: {len(trajectories)}")
    if trajectories:
        lengths = [len(t["actions"]) for t in trajectories]
        rewards = [t["rewards"][-1] for t in trajectories]
        print(f"Game length — mean: {np.mean(lengths):.1f}, max: {np.max(lengths)}, min: {np.min(lengths)}")
        print(f"Results — wins: {sum(r>0 for r in rewards)}, "
              f"draws: {sum(r==0 for r in rewards)}, "
              f"losses: {sum(r<0 for r in rewards)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"\nSaved to {out_path}")


def _year_from_filename(name: str) -> int:
    """Extract year from filenames like 'lichess_elite_2018-06.pgn'."""
    try:
        parts = name.split("_")
        year_part = parts[-1].split("-")[0]
        return int(year_part)
    except (IndexError, ValueError):
        return 0


if __name__ == "__main__":
    main()
