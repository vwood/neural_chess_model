import chess
import random
import tensorflow as tf
import glob
import numpy as np


def pick_random_move(board):
    moves = list(board.generate_legal_moves())
    return random.choice(moves)

piece_to_index = dict([(piece, i) for i, piece in enumerate('rnbqkpPRNBQK')])

# TODO: factor these out into a module
def simple_representation(fen):
    """
    * requires (12, 8, 8) input size

    12 panes, one per type of piece
    """
    result = np.zeros((12, 8, 8), dtype=np.int8)

    flip = False
    if fen.split(' ')[1] == 'b':
        fen = fen.swapcase()
        flip = True
    
    row, col = 0, 0
    for ch in fen:
        if ch == ' ':
            break
        
        if ch == '/':
            row += 1
            col = 0
        elif ch.isdigit():
            col += int(ch)
        else:
            if flip:
                result[piece_to_index[ch], 7 - col, row] = 1
            else:
                result[piece_to_index[ch], col, row] = 1
            col += 1
            
    return result, flip

def run_model(model, board):
    moves = []
    results = []
    
    for move in board.generate_legal_moves():
        moves.append(move)

    for move in moves:
        score = 0.0
        if board.is_capture(move):
            score += 0.1
            
        board.push(move)

        """
        if board.is_check():
            score += 0.1

        if board.is_checkmate():
            score += 0.2

        if board.is_stalemate():
            score -= 0.1

        if board.is_repetition():
            score -= 0.2
            
        if board.is_insufficient_material():
            score -= 0.2
        """
            
        input_, flip = simple_representation(board.fen())
        result = model(np.array([input_]))
        # print(result)
        if flip:
            results.append(np.mean(result) + score)
        else:
            results.append(1.0-(np.mean(result) + score))
        board.pop()

    pick = np.argmax(results)
        
    return moves[pick]


def play_game(fn1, fn2):
    board = chess.Board()

    current = 0
    for i in range(500):
        if board.is_game_over():
            return board.result()
        if i % 2 == 0:
            move = fn1(board)
        else:
            move = fn2(board)
            
        board.push(move)

    return 'timeout'



if __name__ == '__main__':
    all_models = glob.glob('models/*.h5')

    filename1 = 'models/chess-0020-0.009.h5'
    filename2 = 'models/chess-score-0015-0.010.h5'

    model1 = tf.keras.models.load_model(filename1)
    model2 = tf.keras.models.load_model(filename2)

    for i in range(10):
        result = play_game(
            lambda b: run_model(model1, b),
            lambda b: run_model(model2, b)
        )
        print(result)
    
