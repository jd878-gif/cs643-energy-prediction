# CS 643 - PA2: energy consumption prediction container
# Contains: Python 3.11, Java 17, PySpark 3.5.1, the prediction app and the trained model.
FROM python:3.11-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
 && rm -rf /var/lib/apt/lists/* \
 && ln -s "$(dirname "$(dirname "$(readlink -f "$(which java)")")")" /opt/java

ENV JAVA_HOME=/opt/java \
    PYTHONUNBUFFERED=1 \
    SPARK_CONF_DIR=/app/conf \
    MODEL_PATH=/app/model

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY conf/ /app/conf/
COPY prediction/predict.py /app/predict.py
# Trained PipelineModel (copied from S3 after the EMR training job)
COPY model/ /app/model/

# docker run -v /host/dir:/data IMAGE /data/TestDataset.csv
ENTRYPOINT ["python3", "/app/predict.py"]
CMD ["--help"]
