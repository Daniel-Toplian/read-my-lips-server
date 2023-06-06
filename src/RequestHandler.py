import tempfile
from pathlib import Path

import cv2
import tensorflow as tf
from flask import jsonify

from Utils import num_to_char
from Utils import vtt_input_shape
from models.Models import create_vtt_model, create_lc_model

video_to_text_model = None
lips_crop_model = None


class RequestHandler:
    def __init__(self, config):
        weights_file_vtt = config.get('DEFAULT', 'video_to_text_weights_file')
        weights_file_lc = config.get('DEFAULT', 'lips_crop_weights_file')

        self.video_to_text_model = create_vtt_model(vtt_input_shape)
        self.video_to_text_model.load_weights(weights_file_vtt)

        self.lips_crop_model = create_lc_model(weights_file_lc)

    def process_video(self, request):
        if 'video' not in request.files:
            return jsonify({'status': 'error', 'message': 'No video file found in the request.'}), 400

        if video_to_text_model is None:
            return jsonify({'status': 'error', 'message': 'Service in not available at the moment.'}), 500

        uploaded_file = request.files['video']

        video_file = self.preprocessing_input(uploaded_file)
        generated_text = str(self.generate_text(video_file).pop(0).numpy())

        return jsonify({'status': 'success', 'message': 'Video processed successfully!',
                        'generated_text': generated_text}), 200

    def preprocessing_input(self, video):
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
                # frame = crop_mouth_in_frame(frame)
                frames.append(frame[190:236, 80:220, :])

            cap.release()

            mean = tf.math.reduce_mean(frames)
            std = tf.math.reduce_std(tf.cast(frames, tf.float32))

            value = tf.cast((frames - mean), tf.float32) / std
            value = tf.expand_dims(value, axis=0)
            return value

    def generate_text(self, video):
        yhat = self.video_to_text_model.predict(video)
        sequence_length = yhat.shape[1]
        decoded_text = tf.keras.backend.ctc_decode(yhat, input_length=[sequence_length], greedy=True)[0][0].numpy()
        return [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded_text]

    def crop_mouth_in_frame(self, frame):
        ds_factor = 1
        frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mouth_rects = self.lips_crop_model.detectMultiScale(gray, 1.7, 11)

        for (x, y, w, h) in mouth_rects:
            y = int(y - 0.5 * h)
            frame = frame[y:y + h, x:x + w]
            break

        frame_resize = cv2.resize(frame, (75, 50))
        return frame_resize
