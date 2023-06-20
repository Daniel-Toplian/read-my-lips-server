# read-my-lips-server

A flask server written with python 

## Docker

Run with the following command (make sure to replace the variables):

`docker run --rm -it -p $SERVER_PORT:5000 -v "$ABSOLUTE_PATH_TO_MODEL_FILE":/app/resources/model.h5 --name rml-server habani/rml-server`