FROM python:3.9-slim

WORKDIR /backend

COPY . .

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]