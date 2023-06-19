FROM tensorflow/tensorflow

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

WORKDIR /app

COPY . .

ENTRYPOINT [ "python" ]
CMD [ "/app/src/RmlServer.py"]