import configparser

from flask import Flask, request, jsonify
from flask_cors import CORS

from Utils import vtt_input_shape
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

    video_file = request.files['video']
    # video_file = preprocessing_input(request.files['video'])

    generated_text = generate_text(video_file)

    return jsonify({'status': 'success', 'message': 'Video processed successfully!',
                    'generated_text': generated_text}), 200


def preprocessing_input(video):
    # return lips_crop_model.predict(video)
    pass


def generate_text(video):
    return video_to_text_model.predict(video)


if __name__ == '__main__':
    config_server()
    server.run(port=port)
