import io
import os
import tempfile
from pathlib import Path

import cv2
import tensorflow as tf
from autocorrect import Speller
from flask import jsonify

from Utils import num_to_char
from src.models.ModelsCreator import create_lc_model, create_vtt_model

video_to_text_model = None
lips_crop_model = None


def validate_upload_file(uploaded_file):
    allowed_extensions = ['.mp4', '.mov', '.avi', '.mpg']
    allowed_mimetypes = ['video/mp4', 'video/quicktime', 'video/mpeg']

    file_extension = os.path.splitext(uploaded_file.filename)[1]

    return file_extension.lower() not in allowed_extensions or uploaded_file.mimetype not in allowed_mimetypes


class RequestHandler:
    def __init__(self, config):
        weights_file_vtt = config.get('DEFAULT', 'video_to_text_weights_file')
        weights_file_lc = config.get('DEFAULT', 'lips_crop_weights_file')

        self.lips_crop_model = create_lc_model(weights_file_lc)
        self.video_to_text_model = create_vtt_model(weights_file_vtt)
        self.speller = Speller()
        print("----loading complete----")

    def process_video(self, request):
        if 'video' not in request.files or validate_upload_file(request.files['video']):
            return jsonify({'status': 'error', 'message': 'No video file found in the request.'}), 400

        if self.video_to_text_model is None:
            return jsonify({'status': 'error', 'message': 'Service in not available at the moment.'}), 500

        uploaded_file = request.files['video']

        video_chunked_frames, sequence_length = self.preprocessing_input(uploaded_file)

        if video_chunked_frames is not None and sequence_length is not None:
            generated_text = self.generate_text(video_chunked_frames, sequence_length)
            return jsonify({'status': 'success', 'message': 'Video processed successfully!',
                            'generated_text': generated_text}), 200

        return jsonify({'status': 'error', 'message': 'Unable to find a face! Please upload an acceptable video'}), 406

    def preprocessing_input(self, video):
        isNoFace = False
        with tempfile.TemporaryDirectory() as td:
            temp_filename = Path(td) / 'uploaded_video'
            video.save(temp_filename)

            cap = cv2.VideoCapture(str(temp_filename))

            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.crop_mouth_from_face_in_frame(frame)
                if frame is None:
                    isNoFace = True
                    break
                frame = tf.image.rgb_to_grayscale(frame)
                frames.append(frame)

            cap.release()

            if isNoFace:
                return None, None

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
            string_builder.write(sentence.numpy().decode())
        string_builder.write(".")

        generated_text = string_builder.getvalue()
        string_builder.close()
        return self.speller(generated_text)

    def crop_mouth_from_face_in_frame(self, frame):
        face_detected = True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = self.lips_crop_model.detectMultiScale(gray, 1.2, 5)

        if len(face_rects) == 0:
            face_detected = False
        else:
            for (x, y, w, h) in face_rects:
                frame = frame[y:y + h, x:x + w]
                frame_resize = cv2.resize(frame, (140, 140))
                mouth = frame_resize[90:140, 40:100]
                mouth_resize = cv2.resize(mouth, (120, 100))
                break

        if face_detected:
            return mouth_resize
        else:
            return None
