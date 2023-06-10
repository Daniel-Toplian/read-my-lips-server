import tempfile
from pathlib import Path

import io
import cv2
import tensorflow as tf
from flask import jsonify

from Utils import num_to_char
from Utils import vtt_input_shape
from src.models.ModelsCreator import create_lc_model, create_vtt_model

video_to_text_model = None
lips_crop_model = None


class RequestHandler:
    def __init__(self, config):
        weights_file_vtt = config.get('DEFAULT', 'video_to_text_weights_file')
        weights_file_lc = config.get('DEFAULT', 'lips_crop_weights_file')

        self.lips_crop_model = create_lc_model(weights_file_lc)
        self.video_to_text_model = create_vtt_model(weights_file_vtt, input_shape=vtt_input_shape)
        print("----loading complete----")

    def process_video(self, request):
        if 'video' not in request.files:
            return jsonify({'status': 'error', 'message': 'No video file found in the request.'}), 400

        if self.video_to_text_model is None:
            return jsonify({'status': 'error', 'message': 'Service in not available at the moment.'}), 500

        uploaded_file = request.files['video']

        video_chunked_frames, sequence_length = self.preprocessing_input(uploaded_file)
        generated_text = self.generate_text(video_chunked_frames, sequence_length)

        return jsonify({'status': 'success', 'message': 'Video processed successfully!',
                        'generated_text': generated_text}), 200

    def preprocessing_input(self, video):
        with tempfile.TemporaryDirectory() as td:
            temp_filename = Path(td) / 'uploaded_video'
            video.save(temp_filename)

            cap = cv2.VideoCapture(str(temp_filename))

            frames = []
            for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                ret, frame = cap.read()
                frame = self.crop_mouth_in_frame(frame)
                frame = tf.image.rgb_to_grayscale(frame)
                frame = tf.image.resize(frame, [46, 140])
                frames.append(frame)

            cap.release()

            mean = tf.math.reduce_mean(frames)
            std = tf.math.reduce_std(tf.cast(frames, tf.float32))

            value = tf.cast((frames - mean), tf.float32) / std

            # Split frames into chunks of 75 frames
            frame_chunks = [value[i:i + 75] for i in range(0, len(value), 75)]

            # Pad the last chunk if its length is less than 75
            last_chunk_length = len(frame_chunks[-1])
            if last_chunk_length < 75:
                padding = 75 - last_chunk_length
                last_chunk = tf.pad(frame_chunks[-1], paddings=[[0, padding], [0, 0], [0, 0], [0, 0]])
                frame_chunks[-1] = last_chunk

            # Create batch dataset
            dataset = tf.data.Dataset.from_tensor_slices(frame_chunks)
            dataset = dataset.batch(1)

            sequence_length = [chunk.shape[0] for chunk in frame_chunks]

            return dataset, sequence_length

    def generate_text(self, video, sequence_length):
        yhat = self.video_to_text_model.predict(video)
        decoded_text = tf.keras.backend.ctc_decode(yhat, input_length=tf.constant(sequence_length),
                                                   greedy=True)[0][0].numpy()

        string_builder = io.StringIO()
        sentence_list = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded_text]

        for sentence in sentence_list:
            string_builder.write(str(sentence.numpy()))

        generated_text = string_builder.getvalue()
        string_builder.close()
        return generated_text

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
