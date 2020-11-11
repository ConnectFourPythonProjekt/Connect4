from agents.common import BoardPiece, SavedState, PlayerAction
from typing import Optional, Callable, Tuple
import numpy as np

def generate_move_random(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    # TODO: not implemented yet
    action = 3
    return action, saved_state