#!/usr/bin/env python3
"""
CS 643 - Programming Assignment 2
Energy-consumption prediction application (single machine).

Loads the trained Spark PipelineModel, predicts energy consumption for every
row of the test file given on the command line, and prints the RMSE.

Usage:
    python3 predict.py <path/to/TestDataset.csv> [--model ./model] [--output preds.csv]

Inside Docker:
    docker run -v /host/dir:/data <image> /data/TestDataset.csv
"""
import argparse
import os
import sys

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# --------------------------------------------------------------------------
# Schema / loading (identical copy lives in training/train.py)
# --------------------------------------------------------------------------
LABEL = "Energy Consumption"
CATEGORICAL = ["Building Type", "Day of Week"]
NUMERIC = ["Square Footage", "Number of Occupants", "Appliances Used", "Average Temperature"]
_CANONICAL = {c.lower().replace(" ", "").replace("_", ""): c for c in CATEGORICAL + NUMERIC + [LABEL]}


def _normalise(name):
    return name.strip().strip('"').strip("'").lower().replace(" ", "").replace("_", "")


def _detect_delimiter(spark, path):
    header = spark.sparkContext.textFile(path).first()
    for d in [",", ";", "\t", "|"]:
        if d in header:
            return d
    return ","


def load_dataset(spark, path, require_label=True) -> DataFrame:
    """Read a CSV robustly: auto delimiter, quoted/odd header names, type casting."""
    sep = _detect_delimiter(spark, path)
    df = (spark.read.option("header", True).option("sep", sep).option("quote", '"')
          .option("ignoreLeadingWhiteSpace", True).option("ignoreTrailingWhiteSpace", True)
          .csv(path))
    for col in df.columns:
        canon = _CANONICAL.get(_normalise(col))
        if canon and canon != col:
            df = df.withColumnRenamed(col, canon)
    needed = CATEGORICAL + NUMERIC + ([LABEL] if require_label else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; found {df.columns}")
    for c in NUMERIC + ([LABEL] if LABEL in df.columns else []):
        df = df.withColumn(c, F.col(c).cast(DoubleType()))
    for c in CATEGORICAL:
        df = df.withColumn(c, F.trim(F.col(c)))
    return df.dropna(subset=needed)


# --------------------------------------------------------------------------
def default_model_path():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in [os.environ.get("MODEL_PATH", ""), "/app/model", os.path.join(here, "model"),
              os.path.join(here, "..", "model")]:
        if p and os.path.isdir(p):
            return p
    return os.path.join(here, "model")


def main():
    ap = argparse.ArgumentParser(description="Energy consumption prediction - prints RMSE")
    ap.add_argument("test_file", help="Path to the test CSV on the local host")
    ap.add_argument("--model", default=None, help="Path to the saved PipelineModel directory")
    ap.add_argument("--output", default=None, help="Optional CSV path to write predictions")
    args = ap.parse_args()

    test_path = os.path.abspath(args.test_file)
    if not os.path.isfile(test_path):
        sys.exit(f"ERROR: test file not found: {test_path}\n"
                 f"(With Docker, mount its folder with -v, e.g. "
                 f"docker run -v $(pwd):/data IMAGE /data/TestDataset.csv)")
    model_path = os.path.abspath(args.model or default_model_path())
    if not os.path.isdir(model_path):
        sys.exit(f"ERROR: model directory not found: {model_path}")

    spark = (SparkSession.builder.appName("CS643-PA2-EnergyPrediction")
             .master(os.environ.get("SPARK_MASTER", "local[*]"))
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    model = PipelineModel.load("file://" + model_path)
    df = load_dataset(spark, "file://" + test_path, require_label=False)
    preds = model.transform(df).cache()
    n = preds.count()

    print("=" * 60)
    print(f"Model      : {model_path}  ({type(model.stages[-1]).__name__})")
    print(f"Test file  : {test_path}")
    print(f"Rows scored: {n}")

    preds.select(*CATEGORICAL, *NUMERIC,
                 *([LABEL] if LABEL in df.columns else []),
                 F.round("prediction", 2).alias("Predicted Energy")).show(10, truncate=False)

    if args.output:
        import csv
        cols = list(df.columns) + ["Predicted Energy"]
        rows = preds.select(*df.columns, F.col("prediction").alias("Predicted Energy")).collect()
        with open(args.output, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            w.writerows([[r[c] for c in cols] for r in rows])
        print(f"Predictions written to {args.output}")

    if LABEL in df.columns:
        ev = lambda m: RegressionEvaluator(labelCol=LABEL, predictionCol="prediction",
                                           metricName=m).evaluate(preds)
        rmse, mae, r2 = ev("rmse"), ev("mae"), ev("r2")
        print("=" * 60)
        print(f"RMSE = {rmse:.6f}")
        print(f"MAE  = {mae:.6f}")
        print(f"R2   = {r2:.8f}")
        print("=" * 60)
    else:
        print("No 'Energy Consumption' column in the test file -> RMSE cannot be computed.")

    spark.stop()


if __name__ == "__main__":
    main()
