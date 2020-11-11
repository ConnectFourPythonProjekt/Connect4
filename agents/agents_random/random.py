from agents.common import BoardPiece, SavedState, PlayerAction
from typing import Optional, Callable, Tuple
import numpy as np
import random


def generate_move_random(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    actionList = []
    for col in range(board.shape[1]):
        if np.count_nonzero(board[:, col] == 0) > 0:
            actionList.append(col)

    action = random.choice(actionList)
    return action, saved_state