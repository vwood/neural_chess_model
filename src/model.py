import glob
from collections import Counter
import pandas as pd
import numpy as np
import scipy.special as sc
import keras
import tensorflow as tf
import chess

"""
Contains multiple representations for neural networks
All scoring is done from white's point of view, so flipping is necessary

TODO:
* HalfKP structure
 * two halfs
 * king pos * other piece pos
"""

piece_to_index = dict([(piece, i) for i, piece in enumerate('pnbrqkPNBRQK')])

def board_bitmask_to_numpy(mask):
    mask = np.array([mask], dtype=np.uint64).view(np.uint8)
    return np.unpackbits(mask).reshape((8, 8))

class BasicRepresentation:
    """
    * requires (12, 8, 8) input size

    12 panes, one per type of piece
    2 more panes showing squares attacked by that side
    """
    def get_input_size(self):
        return (12, 8, 8)
    
    def from_fen(self, fen):
        result = np.zeros((12, 8, 8), dtype=np.int8)

        board = chess.Board(fen)

        is_flipped = False
        if board.turn == chess.BLACK:
            is_flipped = True
            board = board.mirror()

        for piece in range(1, 7): # chess numbers pieces from 1 to 6
            mask = board.pieces_mask(piece, chess.WHITE)
            result[piece-1, :, :] = board_bitmask_to_numpy(mask)[::-1, ::-1]
            mask = board.pieces_mask(piece, chess.BLACK)
            result[piece+5, :, :] = board_bitmask_to_numpy(mask)[::-1, ::-1]
        
        return result, is_flipped

    
class BasicRepresentationWithAttacked:
    """
    * requires (14, 8, 8) input size

    12 panes, one per type of piece
    2 more panes showing squares attacked by that side
    """
    def get_input_size(self):
        return (14, 8, 8)
    
    def from_fen(self, fen):
        result = np.zeros((14, 8, 8), dtype=np.int8)

        board = chess.Board(fen)

        is_flipped = False
        if board.turn == chess.BLACK:
            is_flipped = True
            board = board.mirror()

        for piece in range(1, 7): # chess numbers pieces from 1 to 6
            mask = board.pieces_mask(piece, chess.WHITE)
            result[piece-1, :, :] = board_bitmask_to_numpy(mask)[::-1, ::-1]
            mask = board.pieces_mask(piece, chess.BLACK)
            result[piece+5, :, :] = board_bitmask_to_numpy(mask)[::-1, ::-1]

        for i in range(8):
            for j in range(8):
                square = (7 - i) + j * 8
                result[12, j, i] = board.attackers_mask(chess.WHITE, square) > 0
                result[13, j, i] = board.attackers_mask(chess.BLACK, square) > 0
        
        return result, is_flipped

    
class CompactRepresentation:
    """
    * requires (6, 8, 8) input size

    7 panes, one per type of piece, +1 -> white, -1 -> black
    """
    def get_input_size(self):
        return (6, 8, 8)
    
    def from_fen(self, fen):
        result = np.zeros((6, 8, 8), dtype=np.int8)

        board = chess.Board(fen)

        is_flipped = False
        if board.turn == chess.BLACK:
            is_flipped = True
            board = board.mirror()

        for piece in range(1, 7): # chess numbers pieces from 1 to 6
            mask = board.pieces_mask(piece, chess.WHITE)
            result[piece-1, :, :] = board_bitmask_to_numpy(mask)[::-1, ::-1]
            mask = board.pieces_mask(piece, chess.BLACK)
            result[piece+5, :, :] = board_bitmask_to_numpy(mask)[::-1, ::-1]
        
        return result, is_flipped

    
class HalfKP:
    """
    * requires (10, 8, 8, 8, 8) input size

    10 panes, one per type of piece

    TODO: requires sparse inputs, a list of classes
    """
    def get_input_size(self):
        return (6, 8, 8)

    def from_fen(fen):
        pass

def create_simple_dense_model(representation):
    input_ = tf.keras.Input(shape=representation.get_input_size(), dtype='int8')
    flat = tf.keras.layers.Flatten()(input_)
    hidden1 = tf.keras.layers.Dense(1024, activation='elu',
                                    #kernel_regularizer=tf.keras.regularizers.L2(1e-9)
                                    )(flat)
    dropout1 = tf.keras.layers.Dropout(0.1)(hidden1)
    hidden2 = tf.keras.layers.Dense(512, activation='elu',
                                    #kernel_regularizer=tf.keras.regularizers.L2(1e-9)
                                    )(dropout1)
    hidden3 = tf.keras.layers.Dense(128, activation='elu',
                                    #kernel_regularizer=tf.keras.regularizers.L2(1e-9)
                                    )(hidden2)

    outputs = tf.keras.layers.Dense(2, activation='sigmoid')(hidden3)
    model = tf.keras.Model(inputs=input_, outputs=outputs)
    model.compile(optimizer="Adam", loss="mse")
    model.summary()
    return model


def create_conv_model(representation):
    input_ = tf.keras.Input(shape=representation.get_input_size(), dtype='float32')

    # permutation solves the problem of having the tensor we want to permute last
    # TODO: solve this problem in the representations.
    perm = tf.keras.layers.Permute((2, 3, 1))(input_)
    conv1 =  tf.keras.layers.Conv1D(filters = 16, kernel_size=(14), padding='same', activation='relu',
                             input_shape=(1, 8,8))(perm)
    # pooling =  tf.keras.layers.MaxPooling1D()(conv1)
    
    flat = tf.keras.layers.Flatten()(conv1)
    hidden1 = tf.keras.layers.Dense(1024, activation='elu',
                                    #kernel_regularizer=tf.keras.regularizers.L2(1e-9)
                                    )(flat)
    dropout1 = tf.keras.layers.Dropout(0.1)(hidden1)
    hidden2 = tf.keras.layers.Dense(512, activation='elu',
                                    #kernel_regularizer=tf.keras.regularizers.L2(1e-9)
                                    )(dropout1)
    hidden3 = tf.keras.layers.Dense(128, activation='elu',
                                    #kernel_regularizer=tf.keras.regularizers.L2(1e-9)
                                    )(hidden2)

    outputs = tf.keras.layers.Dense(2, activation='sigmoid')(hidden3)
    model = tf.keras.Model(inputs=input_, outputs=outputs)
    model.compile(optimizer="Adam", loss="mse")
    model.summary()
    return model

def load_data(filename, representation):
    df = pd.read_csv(filename)

    sample = []
    ys = []
    for i, (fen, wins, total, score) in df.iterrows():
        #if total <= 2:
        #    continue
        if i >= 2000000:
            break

        input_layer, is_flipped = representation.from_fen(fen)
        sample.append(input_layer)
        # score = score * 10
        a, b = wins + score, total - wins + (10 - score)

        # ys.append([sc.btdtri(a+1, b+1, 0.2), sc.btdtri(a+1, b+1, 0.8)])
        if is_flipped:
            ys.append([1.0-score])
        else:
            ys.append([score])
            
    sample = np.array(sample, dtype=np.int8)
    ys = np.array(ys)
            
    return sample, ys


if __name__ == '__main__':
    representation = BasicRepresentationWithAttacked()
    # model = create_simple_dense_model(representation)
    model = create_conv_model(representation)
    sample, ys = load_data('./output/all_scored.csv', representation)

    print(sample.shape, ys.shape)

    valid_size = 2000
    valid_idx = np.random.choice(ys.shape[0], valid_size)
    x_valid = sample[valid_idx]
    y_valid = ys[valid_idx]

    train_idx = [idx for idx in range(ys.shape[0])
                 if idx not in valid_idx]
    x_train = sample[train_idx]
    y_train = ys[train_idx]

    print(x_valid.shape, y_valid.shape, x_train.shape, y_train.shape)
    print(np.mean(ys,axis=0), np.std(ys, axis=0))
    print(sample.shape)

    save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="../models/chess-score-{epoch:04d}-{val_loss:.4f}.h5",
        save_weights_only=False,
        save_best_only=True,
        verbose=1)

    # Train the model with the new callback
    history = model.fit(x_train, y_train,
                        validation_data = (x_valid, y_valid),
                        batch_size=32,
                        epochs=20,
                        callbacks=[save_callback])

    print(history)
