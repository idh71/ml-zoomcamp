FROM --platform=linux/amd64 svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]



# FROM python:3.9.12-slim
# WORKDIR /app
# COPY ["model2.bin", "dv.bin", "./"]

#"predict.py", "model1.bin", "dv.bin",