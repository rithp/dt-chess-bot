"""
dataset_utils.py
Board and move encoding/decoding utilities for the Chess Decision Transformer.

Board state: 768-dim bitboard float vector
  - 12 piece types (6 white + 6 black) × 64 squares
  - 1.0 if the piece is present, 0.0 otherwise

Move encoding: integer in [0, 4095 + 4 promotion extras]
  - Primary: from_sq * 64 + to_sq  (0..4095)
  - Promotions (non-queen) are encoded in a suffix space 4096..4099
    to keep it simple we map all underpromotions to their own slots,
    but default queen-promotions fall naturally in the 0-4095 range via
    the queen bit.

For simplicity and compatibility with the DT integer action-space we use:
  MOVE_VOCAB_SIZE = 4096 + 4  (queen promo = natural index, 4096-4099 for r/b/n promo)
  PADDING_IDX     = 4100      (used to fill masked timesteps)
"""

import chess
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIECE_ORDER = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP,
    chess.ROOK, chess.QUEEN,  chess.KING,
]

MOVE_VOCAB_SIZE = 4096 + 4   # 0-4095 normal/queen-promo, 4096-4099 underpromo (R,B,N per color)
PADDING_IDX     = MOVE_VOCAB_SIZE  # 4100 — used to pad masked steps

# Underpromotion mapping  piece → offset from 4096
_UNDERPROMO_OFFSET = {
    chess.ROOK:   0,
    chess.BISHOP: 1,
    chess.KNIGHT: 2,
}


# ---------------------------------------------------------------------------
# Board → state vector
# ---------------------------------------------------------------------------

def board_to_state(board: chess.Board) -> np.ndarray:
    """
    Encode a chess.Board as a 768-dim float32 bitboard vector.

    Layout: [white_pawns(64), white_knights(64), ..., black_king(64)]
    Always from White's perspective (no board flip).
    """
    state = np.zeros(768, dtype=np.float32)
    for color_idx, color in enumerate([chess.WHITE, chess.BLACK]):
        for piece_idx, piece_type in enumerate(PIECE_ORDER):
            bb = board.pieces(piece_type, color)
            for sq in bb:
                state[color_idx * 384 + piece_idx * 64 + sq] = 1.0
    return state


# ---------------------------------------------------------------------------
# Move → integer index
# ---------------------------------------------------------------------------

def move_to_index(move: chess.Move) -> int:
    """
    Encode a chess.Move as an integer in [0, MOVE_VOCAB_SIZE).

    Queen promotions are folded into the natural from*64+to index (so they
    share the same slot as a non-promotion to that square — but in practice
    a move to the 8th/1st rank from the 7th/2nd must be a promotion).

    Underpromotions (rook, bishop, knight) get dedicated indices 4096-4099.
    """
    from_sq = move.from_square
    to_sq   = move.to_square
    promo   = move.promotion

    if promo is not None and promo != chess.QUEEN:
        return 4096 + _UNDERPROMO_OFFSET[promo]

    return from_sq * 64 + to_sq


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """
    Decode an integer index back to a chess.Move given the current board.

    For the primary 0-4095 range we reconstruct from_sq / to_sq and check
    if a queen promotion is needed.  For 4096-4099 we use the context of
    which pawns can promote to determine from_sq/to_sq.
    """
    if idx < 4096:
        from_sq = idx // 64
        to_sq   = idx %  64
        # Detect if this is a pawn promotion (piece on from_sq is a pawn at rank 7/0)
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            if (piece.color == chess.WHITE and chess.square_rank(to_sq) == 7) or \
               (piece.color == chess.BLACK and chess.square_rank(to_sq) == 0):
                return chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        return chess.Move(from_sq, to_sq)

    else:
        # Underpromotion: find the first legal pawn promotion move of this type
        promo_type = [chess.ROOK, chess.BISHOP, chess.KNIGHT][idx - 4096]
        for move in board.legal_moves:
            if move.promotion == promo_type:
                return move
        # Fallback: return any legal move
        return next(iter(board.legal_moves))


# ---------------------------------------------------------------------------
# Legal move mask
# ---------------------------------------------------------------------------

def legal_move_mask(board: chess.Board) -> np.ndarray:
    """
    Return a boolean mask of shape (MOVE_VOCAB_SIZE,) where True means the
    move is legal in the current position.
    """
    mask = np.zeros(MOVE_VOCAB_SIZE, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_index(move)] = True
    return mask


# ---------------------------------------------------------------------------
# Game outcome → reward
# ---------------------------------------------------------------------------

def outcome_to_reward(result: str, color: chess.Color) -> float:
    """
    Convert a PGN result string to a sparse reward from the given player's
    perspective.
      '1-0' → White wins
      '0-1' → Black wins
      '1/2-1/2' → draw
    """
    if result == '1-0':
        return 1.0 if color == chess.WHITE else -1.0
    elif result == '0-1':
        return 1.0 if color == chess.BLACK else -1.0
    elif result == '1/2-1/2':
        return 0.0
    else:
        return 0.0  # unknown / abandoned
