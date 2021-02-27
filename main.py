from concurrent.futures._base import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
import numpy as np
from typing import Optional, Callable

from agents.common import PlayerAction, BoardPiece, SavedState, GenMove
from agents import common
from agents.agents_random import generate_move
from agents.agents_minimax import minimax
from agents.agents_minimax import minimax_gen_move
from agents.agents_montecarlo.monte_carlo import Node, generate_move_montecarlo, get_position_mask_bitmap, \
    connected_four, connected_three, connected_two, number_of_connected
from agents.agents_montecarlo_heuristic.monte_carlo_heuristic import generate_move_MCH


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


def play():
    board_copy = common.initialize_game_state()
    while True:
        move, ss = generate_move_montecarlo(board_copy, BoardPiece(1), None)
        common.apply_player_action(board_copy, move, BoardPiece(1), False)
        p, m = get_position_mask_bitmap(1, board_copy)
        if m == 562949953421311:
            return 0
        if connected_four(p):
            return 1
        random, sss = generate_move_MCH(board_copy, BoardPiece(2), None)
        common.apply_player_action(board_copy, random, BoardPiece(2), False)
        p1, m1 = get_position_mask_bitmap(2, board_copy)
        if connected_four(p1):
            return -1
        if m1 == 562949953421311:
            return 0


if __name__ == '__main__':
    #human_vs_agent(generate_move_montecarlo)
    # root = Node()
    board = np.zeros((6, 7))
    board[0, 0:7] = [0, 2, 1, 1, 2, 0, 0]
    board[1, 0:7] = [0, 0, 2, 1, 1, 0, 0]
    board[2, 0:7] = [0, 0, 0, 2, 2, 0, 0]
    board[3, 0:7] = [0, 0, 0, 0, 0, 0, 0]
    board[4, 0:7] = [0, 0, 0, 0, 0, 0, 0]
    board[5, 0:7] = [0, 0, 0, 0, 0, 0, 0]
    print(common.pretty_print_board(board))
    op_p, m = get_position_mask_bitmap(2,board)
    # print(connected_two(op_p,m))
    print(generate_move_montecarlo(board, 1, None))




    # with ThreadPoolExecutor(max_workers=8) as executor:
    #     win_1 = 0
    #     win_2 = 0
    #     draw = 0
    #     for finished_game in as_completed([
    #         executor.submit(play)
    #         for i in range(8)
    #     ]):
    #         result = finished_game.result()
    #         print(result)
    #         if result == 1:
    #             win_1 += 1
    #         if result == -1:
    #             win_2 += 1
    #         if result == 0:
    #             draw += 1
    #
    #     print("Winns for pure Monte Carlo agent: ", win_1)
    #     print("Winns for  heuristic Monte Carlo agent: ", win_2)
    #     print("Draws: ", draw)
