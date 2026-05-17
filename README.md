# Chess Decision Transformer

A chess-playing bot trained using the [Decision Transformer](https://github.com/kzl/decision-transformer) architecture on the [Lichess Elite Database](https://database.lichess.org/) (2200+ vs 2400+ ELO games).

## Architecture

```
[RTG_t | board_t | move_t] × K timesteps
         ↓
   GPT-2 backbone (causal transformer)
         ↓
   Linear head → logits over 4096 move indices
         ↓
   mask illegal moves → argmax → UCI move
```

| Component | Detail |
|---|---|
| Board state | 768-dim bitboard (12 piece planes × 64 squares) |
| Action space | Integer in [0, 4095] — `from_sq × 64 + to_sq` |
| Reward | Sparse: +1 win, 0 draw, -1 loss (terminal step only) |
| Loss | Cross-entropy over move logits |
| Context K | 20 moves (configurable) |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Step 1 — Parse PGN Data

```bash
python data/parse_pgn.py \
    --pgn_dir "/Users/rithvikp/Downloads/Lichess Elite Database" \
    --out data/chess_trajectories.pkl \
    --max_games 50000 \
    --min_year 2018
```

| Flag | Default | Description |
|---|---|---|
| `--pgn_dir` | — | Path to Lichess Elite PGN directory |
| `--out` | `data/chess_trajectories.pkl` | Output pickle path |
| `--max_games` | 50000 | Max games to parse |
| `--min_year` | 0 | Only load PGNs from this year onwards |

---

## Step 2 — Train

```bash
python train_chess.py \
    --data_path data/chess_trajectories.pkl \
    --K 20 \
    --embed_dim 128 \
    --n_layer 3 \
    --n_head 4 \
    --batch_size 64 \
    --max_iters 10 \
    --num_steps_per_iter 5000 \
    --device cuda
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--K` | 20 | Context window (moves) |
| `--embed_dim` | 128 | GPT-2 hidden size |
| `--n_layer` | 3 | Number of transformer layers |
| `--n_head` | 4 | Attention heads |
| `--target_rtg` | 1.0 | RTG to condition on at eval (1.0 = aim to win) |
| `--num_eval_games` | 20 | Games vs random mover per eval |
| `--log_to_wandb` | False | Enable W&B logging |

Checkpoints are saved to `checkpoints/chess_dt.pt` whenever the win rate improves.

---

## Project Structure

```
chessbot/
├── data/
│   ├── dataset_utils.py       # Board/move encoding
│   └── parse_pgn.py           # PGN → trajectory pkl
├── chess_dt/
│   ├── models/
│   │   ├── trajectory_gpt2.py           # GPT-2 backbone
│   │   └── chess_decision_transformer.py # Chess DT model
│   ├── training/
│   │   └── chess_seq_trainer.py
│   └── evaluation/
│       └── evaluate_chess.py
├── train_chess.py             # Main training script
├── requirements.txt
└── README.md
```

---

## Design Decisions

**Why sparse rewards?**  
The DT conditions on Return-to-Go (RTG). With sparse outcome rewards (+1/-1/0), conditioning on RTG=+1 at inference steers the model toward moves observed in winning games — a direct analogy to the original DT paper.

**Why cross-entropy instead of MSE?**  
Chess has a discrete action space. Cross-entropy over the move vocabulary is the natural loss for classification over ~4096 possible moves.

**Why bitboards?**  
768-dim float vectors are fast to compute, unambiguous, and map cleanly to the DT's linear state embedding — no tokenization overhead.
