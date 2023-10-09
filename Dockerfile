FROM python:3.10-slim

WORKDIR /app

COPY ./requirements.txt .

COPY ./ml/ .

RUN pip3 install -r requirements.txt --no-cache-dir

CMD ["uvicorn", "main:main_app", "--host", "127.0.0.1", "--port", "9000"]
