import chess.pgn
import glob
import bz2
import io
import os
from collections import Counter
import csv

"""
Convert pgn data to a csv format

Ignore computer games, as we want a human-like model
"""

result_table = {'1-0': (1, 0),
                '0-1': (0, 1),
                '1/2-1/2': (0.5, 0.5)}

bad_ends = set([
    'Black forfeits on time',
    'Black forfeits by disconnection',
    'White forfeits on time',
    'White forfeits by disconnection'])

def stored_fen(fen):
    return ' '.join(fen.split(' ')[:4])

def convert_file(filename, limit=8000, min_count=2):
    """
    Convert pgn games into individual position records for a csv
    filename - the pgn file to read
    limit - the maximum number of games to examine
    min_count - number of times a position has to occur to be counted

    creates a csv file in the output directory containing:
       the fen representation, white wins and total games
    """
    results = Counter()
    boards_w = Counter()
    boards_total = Counter()
    analysed_games = 0

    with bz2.open(filename, "rb") as zip:
        pgn = io.StringIO(zip.read().decode('ascii'))

        while game := chess.pgn.read_game(pgn):
            if analysed_games > limit:
                break

            if (game.headers.get('BlackIsComp', 'No') == 'Yes'
                or game.headers.get('WhiteIsComp', 'No') == 'Yes'):
                continue

            end = game.end().comment
            if end in bad_ends:
                continue

            results[end] += 1

            analysed_games += 1

            score = result_table[game.headers['Result']] 

            board = game.board()
            for move in game.mainline_moves():
                board.push(move)

                fen = stored_fen(board.fen())
                boards_w[fen] += score[0]
                boards_total[fen] += 1
                
    positions = 0            
    for k, v in boards_total.items():
        if v >= min_count:
            positions += 1
    print(f"found {positions} positions occurring at least {min_count} times")

    filename = os.path.splitext(os.path.basename(filename))[0]
    with open(f'output/{filename}.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['fen', 'wins', 'total'])
        for board, total in boards_total.items():
            wins = boards_w[board]
            writer.writerow([board, wins, total])


if __name__ == '__main__':
    for filename in glob.glob("data/*.bz2"):
        convert_file(filename)

    print(len(boards_total))
