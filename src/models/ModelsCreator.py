import cv2
import keras
import tf


def create_lc_model(weight_file):
    face_cascade = cv2.CascadeClassifier(weight_file)

    return face_cascade


def create_vtt_model(weights_file):
    return keras.models.load_model(weights_file, custom_objects={'CTCLoss': CTCLoss})


def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss