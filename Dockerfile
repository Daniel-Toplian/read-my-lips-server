FROM tensorflow/tensorflow

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

WORKDIR /app

COPY src .
COPY resources resources
COPY env.properties .

ENTRYPOINT [ "python" ]
CMD [ "RmlServer.py"]