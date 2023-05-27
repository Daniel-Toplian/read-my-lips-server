import configparser
import tempfile
from pathlib import Path

import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

from Utils import vtt_input_shape, num_to_char
from models.Models import create_vtt_model

server = Flask(__name__)
CORS(server)

video_to_text_model = None
lips_crop_model = None
port = 5000  # Default port


def config_server():
    global port
    global video_to_text_model
    global lips_crop_model

    config = configparser.ConfigParser()
    config.read('env.properties')

    port = config.get('DEFAULT', 'server_port')
    weights_file_vtt = config.get('DEFAULT', 'video_to_text_weights_file')
    weights_file_lc = config.get('DEFAULT', 'lips_crop_weights_file')

    video_to_text_model = create_vtt_model(vtt_input_shape)
    video_to_text_model.load_weights(weights_file_vtt)

    # lips_crop_model = create_lc_model()
    # lips_crop_model.load_model(weights_file_lc)


@server.route('/video-to-text', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file found in the request.'}), 400

    if video_to_text_model is None:
        return jsonify({'status': 'error', 'message': 'Service in not available at the moment.'}), 500

    uploaded_file = request.files['video']
    video_file = preprocessing_input(uploaded_file)

    generated_text = generate_text(video_file)

    return jsonify({'status': 'success', 'message': 'Video processed successfully!',
                    'generated_text': generated_text}), 200


def preprocessing_input(video):
    with tempfile.TemporaryDirectory() as td:
        temp_filename = Path(td) / 'uploaded_video'
        video.save(temp_filename)

        cap = cv2.VideoCapture(str(temp_filename))

        # Preprocessing
        frames = []
        for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read()
            resized_frame = cv2.resize(frame, (360, 288))
            frame = tf.image.rgb_to_grayscale(resized_frame)
            frames.append(frame[190:236, 80:220, :])

        cap.release()

        mean = tf.math.reduce_mean(frames)
        std = tf.math.reduce_std(tf.cast(frames, tf.float32))

        return tf.cast((frames - mean), tf.float32) / std
        # return lips_crop_model.predict(video)


def generate_text(video):
    yhat = video_to_text_model.predict(video)
    decoded_text = tf.keras.backend.ctc_decode(yhat, input_length=[75, 75], greedy=True)[0][0].numpy()
    return [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded_text]


if __name__ == '__main__':
    config_server()
    server.run(port=port)
