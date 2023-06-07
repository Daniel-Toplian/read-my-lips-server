import pickle

from keras import Sequential
from keras.layers import Conv3D, Activation, MaxPool3D, TimeDistributed, Flatten, Bidirectional, LSTM, Dropout, Dense
from keras.optimizers import Adam
import cv2
from tensorflow.python.keras.models import load_model

from src.Utils import char_to_num


# new model
# def create_vtt_model(input_shape, learning_rate=0.0001):
#     model = Sequential()
#     # Three layers of 3D/spatiotemporal convolutions.
#     model.add(Conv3D(128, 3, input_shape=input_shape, padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool3D((1, 2, 2)))
#
#     model.add(Conv3D(256, 3, padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool3D((1, 2, 2)))
#
#     model.add(Conv3D(75, 3, padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool3D((1, 2, 2)))
#
#     # Flattens each time slice independently.
#     model.add(TimeDistributed(Flatten()))
#
#     # Two layers of Bi-LSTM's. return_sequences=True makes the network output a sequence of predictions,
#     # one for each time step of the input sequence.
#     model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
#     model.add(Dropout(.5))
#
#     model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
#     model.add(Dropout(.5))
#
#     # Linear transformation (dense layer) and output (softmax layer).
#     model.add(Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'))
#     model.compile(optimizer=Adam(learning_rate=learning_rate))
#
#     return model

# class ModelsCreator:


def create_lc_model(weight_file):
    mouth_cascade = cv2.CascadeClassifier(weight_file)

    return mouth_cascade


# old model
def create_vtt_model(weights_file, input_shape, learning_rate=0.0001):
    model = Sequential()
    model.add(Conv3D(128, 3, input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=learning_rate))

    model.load_weights(weights_file)

    return model
