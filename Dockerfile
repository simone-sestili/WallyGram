FROM python:3.9

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY ./app /app

RUN python dataloader.py

RUN python train.py

COPY ./app /app

CMD ["python", "main.py"]