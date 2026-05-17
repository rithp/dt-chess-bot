"""
chess_seq_trainer.py
Sequence trainer adapted for the Chess Decision Transformer.

Key difference from the original SequenceTrainer:
  - Loss: cross-entropy over move logits (not MSE over continuous actions)
  - Actions are integer indices — we use the attention_mask to ignore padded steps
"""

import numpy as np
import torch
import torch.nn.functional as F


class ChessSequenceTrainer:
    """
    Trains the ChessDecisionTransformer via teacher-forced cross-entropy loss.

    get_batch() must return:
        states        (B, K, 768)  float32
        actions       (B, K)       int64
        rewards       (B, K, 1)    float32   (not used in loss)
        dones         (B, K)       bool      (not used in loss)
        rtg           (B, K+1, 1)  float32   — we use rtg[:, :-1, :]
        timesteps     (B, K)       int64
        attention_mask(B, K)       float32   0/1
    """

    def __init__(
        self,
        model,
        optimizer,
        batch_size: int,
        get_batch,
        scheduler=None,
        eval_fns=None,
    ):
        self.model      = model
        self.optimizer  = optimizer
        self.batch_size = batch_size
        self.get_batch  = get_batch
        self.scheduler  = scheduler
        self.eval_fns   = eval_fns or []

        self.diagnostics = {}
        self.start_time  = None

    # -------------------------------------------------------------------------

    def train_step(self) -> float:
        """Single gradient-update step. Returns the scalar loss."""
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)

        # actions shape: (B, K)  — teacher-forced targets
        action_target = actions.clone()   # (B, K)

        # Forward pass
        action_logits = self.model.forward(
            states,
            actions,
            rtg[:, :-1],          # RTG up to but not including terminal step
            timesteps,
            attention_mask=attention_mask,
        )  # (B, K, MOVE_VOCAB_SIZE)

        # Flatten for cross-entropy, ignoring padded positions
        B, K, V = action_logits.shape
        mask_flat    = attention_mask.reshape(-1).bool()         # (B*K,)
        logits_flat  = action_logits.reshape(B * K, V)[mask_flat]  # (N, V)
        targets_flat = action_target.reshape(B * K)[mask_flat]    # (N,)

        loss = F.cross_entropy(logits_flat, targets_flat)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        with torch.no_grad():
            # Top-1 accuracy on non-padded steps
            preds = logits_flat.argmax(dim=-1)
            acc   = (preds == targets_flat).float().mean().item()
            self.diagnostics["training/move_accuracy"] = acc
            self.diagnostics["training/loss"]          = loss.item()

        return loss.item()

    # -------------------------------------------------------------------------

    def train_iteration(self, num_steps: int, iter_num: int = 0, print_logs: bool = False) -> dict:
        """Run num_steps training steps, then evaluate."""
        import time
        self.start_time = time.time()

        train_losses = []
        train_start  = time.time()

        self.model.train()
        for _ in range(num_steps):
            loss = self.train_step()
            train_losses.append(loss)

        self.model.eval()
        eval_start = time.time()

        outputs = {
            "time/training":    time.time() - train_start,
            "training/loss_mean": np.mean(train_losses),
            "training/loss_std":  np.std(train_losses),
        }
        outputs.update(self.diagnostics)

        for eval_fn in self.eval_fns:
            outputs.update(eval_fn(self.model))

        outputs["time/total"] = time.time() - self.start_time

        if print_logs:
            print("=" * 80)
            print(f"Iteration {iter_num}")
            for k, v in sorted(outputs.items()):
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            print("=" * 80)

        return outputs
