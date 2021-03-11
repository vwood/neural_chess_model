import glob
from collections import Counter
import pandas as pd
import numpy as np
import scipy.special as sc
import keras
import tensorflow as tf

"""
Contains multiple representations for neural networks
All scoring is done from white's point of view, so flipping is necessary

TODO:
* HalfKP structure
 * two halfs
 * king pos * other piece pos
"""


piece_to_index = dict([(piece, i) for i, piece in enumerate('rnbqkpPRNBQK')])

def simple_representation(fen):
    """
    * requires (12, 8, 8) input size

    12 panes, one per type of piece
    """
    result = np.zeros((12, 8, 8), dtype=np.int8)

    is_flipped = False
    if fen.split(' ')[1] == 'b':
        fen = fen.swapcase()
        is_flipped = True
    
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
            if is_flipped:
                result[piece_to_index[ch], 7 - col, row] = 1
            else:
                result[piece_to_index[ch], col, row] = 1
            col += 1
            
    return result, is_flipped

def simple_compact_representation(fen):
    """
    * requires (6, 8, 8) input size

    7 panes, one per type of piece, +1 -> white, -1 -> black
    """
    result = np.zeros((6, 8, 8), dtype=np.int8)

    # Check if we need to flip the board
    is_flipped = False
    if fen.split(' ')[1] == 'b':
        fen = fen.swapcase()
        is_flipped = True
    
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
            if is_flipped:
                if ch.isupper():
                    result[piece_to_index[ch % 6], 7-col, row] = 1
                else:
                    result[piece_to_index[ch % 6], 7-col, row] = -1
            else:
                if ch.isupper():
                    result[piece_to_index[ch % 6], col, row] = -1
                else:
                    result[piece_to_index[ch % 6], col, row] = 1
            col += 1
            
    return result, is_flipped

def create_model():
    input_ = tf.keras.Input(shape=(12,8,8,), dtype='int8')
    flat = tf.keras.layers.Flatten()(input_)
    hidden1 = tf.keras.layers.Dense(768, activation='elu',
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
    return model

def load_data(filename, representation_fn)
    df = pd.read_csv(filename)

    sample = []
    ys = []
    for i, (fen, wins, total, score) in df.iterrows():
        #if total <= 2:
        #    continue
        if i >= 2000000:
            break

        representation, is_flipped = representation_fn(fen)
        sample.append(representation)
        # score = score * 10
        a, b = wins + score, total - wins + (10 - score)

        # ys.append([sc.btdtri(a+1, b+1, 0.2), sc.btdtri(a+1, b+1, 0.8)])
        if flip:
            ys.append([1.0-score])
        else:
            ys.append([score])
            
    sample = np.array(sample, dtype=np.int8)
    ys = np.array(ys)
            
    return sample, ys


if __name__ == '__main__':
    model = create_model()
    sample, ys = load_data('output/all_scored.csv', simple_representation)

    print(sample.shape, ys.shape)

    valid_size = 10000
    valid_idx = np.random.choice(ys.shape[0], valid_size)
    x_valid = sample[valid_idx]
    y_valid = ys[valid_idx]

    train_idx = [idx for idx in range(ys.shape[0])
                 if idx not in valid_idx]
    x_train = sample[train_idx]
    y_train = ys[train_idx]

    print(x_valid.shape, y_valid.shape, x_train.shape, y_train.shape)
    print(np.mean(ys,axis=0), np.std(ys, axis=0))
    print(df.total.mean())
    print(df.shape)
    print(sample.shape)

    save_callback = tf.keras.callbacks.ModelCheckpoint(filepath="models/chess-score-{epoch:04d}-{val_loss:.3f}.h5",
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
