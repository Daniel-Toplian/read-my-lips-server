import configparser

from flask import Flask
from flask_cors import CORS
from flask import request

from RequestHandler import RequestHandler

server = Flask(__name__)
CORS(server)

port = 5000  # Default port
request_handler = None


def config_server():
    global port
    global request_handler

    config = configparser.ConfigParser()
    config.read('env.properties')

    port = config.get('DEFAULT', 'server_port')
    request_handler = RequestHandler(config)


@server.route('/video-to-text', methods=['POST'])
def process_video():
    return request_handler.process_video(request)


if __name__ == '__main__':
    config_server()
    from waitress import serve
    serve(server, host="0.0.0.0", port=port)
