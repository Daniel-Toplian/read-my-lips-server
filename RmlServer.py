from flask import Flask, request, jsonify
import pickle
import configparser

server = Flask(__name__)

video_to_text_model = None
lips_crop_model = None
port = 5000


def load_model(file_name, loaded_model):
    with open(file_name, 'rb') as f:
        loaded_model = pickle.load(f)


def config_server():
    global port
    global video_to_text_model
    global lips_crop_model

    config = configparser.ConfigParser()
    config.read('env.properties')

    weights_file_vtt = config.get('DEFAULT', 'video_to_text_weights_file')
    weights_file_lc = config.get('DEFAULT', 'lips_crop_weights_file')
    port = config.get('DEFAULT', 'server_port')

    # load_model(weights_file_vtt, video_to_text_model)
    # load_model(weights_file_lc, lips_crop_model)


@server.route('/video-to-text', methods=['POST'])
def process_video():
    if video_to_text_model is None or 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'Model not loaded. Please load the model first.'}), 500

    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file found in the request.'}), 400

    video_file = preprocessing_input(request.files['video'])

    generated_text = generate_text(video_file)

    return jsonify({'status': 'success', 'message': 'Video processed successfully!',
                    'generated_text': generated_text}), 200


def preprocessing_input(video):
    return lips_crop_model.predict(video)


def generate_text(video):
    return video_to_text_model.predict(video)


if __name__ == '__main__':
    config_server()
    server.run(port=port)
