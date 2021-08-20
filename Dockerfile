FROM tensorflow/tensorflow 

WORKDIR /code

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src/ .
COPY models/ models/.
COPY res/ res/.

ENTRYPOINT ["python", "./tefe.py"]
