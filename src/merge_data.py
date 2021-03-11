import glob
from collections import Counter
import pandas as pd
import csv
import chess
import tqdm

"""
Score positions with a chess engine
"""

def score_positions(filenames, output_filename, engine=None, min_count=2, engine_depth=5):
    """
    Accumulates scores for positions,
    optionally uses an engine to also score positions

    outputs a csv file containing fen positions, wins, total games, and engine score
    """

    # accumulate scores for positions
    wins = Counter()
    totals = Counter()

    for filename in filenames:
        print(f"Reading from {filename}")
        df = pd.read_csv(filename)

        with tqdm(total = len(df)) as progress_bar:
            for i, (fen, w, t) in df.iterrows():
                progress_bar.update(1)
                wins[fen] += w
                totals[fen] += t

    # write output
    with open(output_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['fen', 'wins', 'total', 'score'])

        print(f"Writing output to {output_filename}")
        
        score = 0
        with tqdm(total = len(totals)) as progress_bar:
            for i, (fen, t) in enumerate(totals.items()):
                progress_bar.update(1)

                if t < min_count:
                    continue

                if engine is not None:
                    board = chess.Board(fen)
                    info = engine.analyse(board, chess.engine.Limit(depth=engine_depth))
                    score = info["score"].white().score(mate_score=1000)
                    score = 1.0 / (1.0 + np.exp(-score / 100))

                time.sleep(0.01)
                w = wins[fen]
                writer.writerow([fen, w, t, score])

                
if __name__ == '__main__':
    engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    filenames = glob.glob("output/*standard*.csv")
    output_filename = 'output/all_scored_2.csv'

    score_positions(filenames, output_filename, engine=engine)

                
