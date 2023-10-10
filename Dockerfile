FROM python:3.10-slim

WORKDIR /app

COPY ./requirements.txt .

COPY ./ml/ .

RUN pip3 install -r requirements.txt --no-cache-dir

CMD ["uvicorn", "main:main_app", "--host", "0.0.0.0", "--port", "9000"]
