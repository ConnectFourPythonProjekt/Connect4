import numpy as np
from typing import Optional, Callable

from agents.common import PlayerAction, BoardPiece, SavedState, GenMove
from agents import common
from agents.agents_random import generate_move
from agents.agents_minimax import minimax
from agents.agents_minimax import minimax_gen_move
from agents.agents_montecarlo.monte_carlo import Node, generate_move_montecarlo, get_position_mask_bitmap


def user_move(board: np.ndarray, _player: BoardPiece, saved_state: Optional[SavedState]):
    action = PlayerAction(-1)
    while not 0 <= action < board.shape[1]:
        try:
            action = PlayerAction(input("Column? "))
        except:
            pass
    return action, saved_state


def human_vs_agent(
    generate_move_1: GenMove,
    generate_move_2: GenMove = user_move,
    player_1: str = "Player 1",
    player_2: str = "Player 2",
    args_1: tuple = (),
    args_2: tuple = (),
    init_1: Callable = lambda board, player: None,
    init_2: Callable = lambda board, player: None,
):
    import time
    from agents.common import PLAYER1, PLAYER2, GameState
    from agents.common import initialize_game_state, pretty_print_board, apply_player_action, check_end_state

    players = (PLAYER1, PLAYER2)
    for play_first in (1, -1):
        for init, player in zip((init_1, init_2)[::play_first], players):
            init(initialize_game_state(), player)

        saved_state = {PLAYER1: None, PLAYER2: None}
        board = initialize_game_state()
        gen_moves = (generate_move_1, generate_move_2)[::play_first]
        player_names = (player_1, player_2)[::play_first]
        gen_args = (args_1, args_2)[::play_first]

        playing = True
        while playing:
            for player, player_name, gen_move, args in zip(
                players, player_names, gen_moves, gen_args,
            ):
                t0 = time.time()
                print(pretty_print_board(board))
                print(
                    f'{player_name} you are playing with {"X" if player == PLAYER1 else "O"}'
                )
                action, saved_state[player] = gen_move(
                    board.copy(), player, saved_state[player], *args
                )
                print(f"Move time: {time.time() - t0:.3f}s")
                apply_player_action(board, action, player)
                end_state = check_end_state(board, player)
                if end_state != GameState.STILL_PLAYING:
                    print(pretty_print_board(board))
                    if end_state == GameState.IS_DRAW:
                        print("Game ended in draw")
                    else:
                        print(
                            f'{player_name} won playing {"X" if player == PLAYER1 else "O"}'
                        )
                    playing = False
                    break



if __name__ == '__main__':
    # human_vs_agent(generate_move_montecarlo)
    board = np.zeros((6, 7))
    board[0, 0:7] = [1, 2, 2, 1, 2, 1, 1]
    board[1, 0:7] = [1, 2, 0, 1, 2, 0, 0]
    board[2, 0:7] = [0, 2, 0, 2, 0, 0, 0]
    board[3, 0:7] = [0, 0, 0, 1, 0, 0, 0]
    board[4, 0:7] = [0, 0, 0, 0, 0, 0, 0]
    board[5, 0:7] = [0, 0, 0, 0, 0, 0, 0]

    print(common.pretty_print_board(board))
    print(generate_move_montecarlo(board, BoardPiece(1), None))
