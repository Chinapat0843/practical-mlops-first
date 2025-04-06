FROM python:3.11.4

ARG VERSION
LABEL org.label-schema.version=$VERSION

COPY ./requirements.txt /webapp/requirements.txt
WORKDIR /webapp

RUN pip install -r requirements.txt

COPY webapp/ /webapp/

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
