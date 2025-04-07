FROM python:3.11

# ป้องกัน numpy 2.x โดย fix เวอร์ชัน
COPY requirements.txt /webapp/requirements.txt
WORKDIR /webapp

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y curl && \
    curl -o distilroberta-sequence-classification.onnx https://storage.cloud.google.com/ml-model-artifacts-first/distilroberta-sequence-classification.onnx

COPY webapp/ /webapp/

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "app:app", "--timeout", "300", "--preload"]

