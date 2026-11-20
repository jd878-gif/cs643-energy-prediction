#!/usr/bin/env python3
"""
CS 643 - Programming Assignment 2
Parallel training of an energy-consumption regression model with Spark MLlib.

Runs on an EMR Spark cluster (1 primary + 3 core = 4 EC2 instances), but also
works locally (`--master local[*]`) for quick testing.

What it does
  1. Loads TrainingDataset.csv and ValidationDataset.csv (local path, HDFS or S3).
  2. Builds a feature pipeline: StringIndexer -> OneHotEncoder -> VectorAssembler.
  3. Trains several MLlib regressors over a hyper-parameter grid
     (Linear Regression, Decision Tree, Random Forest, Gradient-Boosted Trees).
     Every fit is a distributed Spark job executed by the executors on all
     worker nodes.
  4. Scores every candidate on the validation set (RMSE, MAE, R2) and keeps the
     one with the lowest validation RMSE.
  5. Saves the winning PipelineModel (features + regressor) to --model-output,
     and a metrics report to --metrics-output.

Example (EMR):
  spark-submit --deploy-mode cluster s3://BUCKET/code/train.py \
      --train s3://BUCKET/data/TrainingDataset.csv \
      --validation s3://BUCKET/data/ValidationDataset.csv \
      --model-output s3://BUCKET/model \
      --metrics-output s3://BUCKET/metrics
"""
import argparse
import itertools
import json
import time

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import (DecisionTreeRegressor, GBTRegressor,
                                   LinearRegression, RandomForestRegressor)
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# --------------------------------------------------------------------------
# Schema / loading (identical copy lives in prediction/predict.py)
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
# Model definition
# --------------------------------------------------------------------------
def feature_stages():
    idx = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in CATEGORICAL]
    ohe = OneHotEncoder(inputCols=[f"{c}_idx" for c in CATEGORICAL],
                        outputCols=[f"{c}_ohe" for c in CATEGORICAL],
                        handleInvalid="keep")
    asm = VectorAssembler(inputCols=NUMERIC + [f"{c}_ohe" for c in CATEGORICAL],
                          outputCol="features", handleInvalid="skip")
    return idx + [ohe, asm]


def grid(**params):
    keys = list(params)
    for values in itertools.product(*params.values()):
        yield dict(zip(keys, values))


def candidates(quick=False):
    """(name, estimator) pairs - the hyper-parameter search space."""
    out = []
    # 1) Linear regression (the "simple model" the assignment asks to start with)
    for p in grid(regParam=[0.0, 0.001, 0.01, 0.1, 1.0], elasticNetParam=[0.0, 0.5, 1.0]):
        if p["regParam"] == 0.0 and p["elasticNetParam"] != 0.0:
            continue  # elastic-net mix is meaningless without regularisation
        out.append((f"LinearRegression{p}",
                    LinearRegression(featuresCol="features", labelCol=LABEL,
                                     maxIter=1000, tol=1e-12, **p)))
    if quick:
        return out
    # 2) Tree-based models for comparison
    for p in grid(maxDepth=[5, 10, 15]):
        out.append((f"DecisionTree{p}",
                    DecisionTreeRegressor(featuresCol="features", labelCol=LABEL, seed=42, **p)))
    for p in grid(numTrees=[50, 150], maxDepth=[8, 12]):
        out.append((f"RandomForest{p}",
                    RandomForestRegressor(featuresCol="features", labelCol=LABEL, seed=42, **p)))
    for p in grid(maxIter=[100, 200], maxDepth=[3, 5], stepSize=[0.1]):
        out.append((f"GBT{p}",
                    GBTRegressor(featuresCol="features", labelCol=LABEL, seed=42, **p)))
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Parallel training of energy consumption model")
    ap.add_argument("--train", required=True, help="TrainingDataset.csv (local, hdfs:// or s3://)")
    ap.add_argument("--validation", required=True, help="ValidationDataset.csv")
    ap.add_argument("--model-output", required=True, help="Where to save the best PipelineModel")
    ap.add_argument("--metrics-output", default=None, help="Where to save metrics JSON (optional)")
    ap.add_argument("--partitions", type=int, default=0,
                    help="Repartition training data (default = cluster default parallelism)")
    ap.add_argument("--quick", action="store_true", help="Only search linear models")
    args = ap.parse_args()

    spark = SparkSession.builder.appName("CS643-PA2-EnergyTraining").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    # ---- Cluster info (evidence of parallel execution) ----
    parts = args.partitions or max(sc.defaultParallelism, 4)
    # Run a tiny distributed job and record which machines executed its tasks.
    hosts = sorted(set(sc.parallelize(range(parts * 8), parts * 4)
                       .map(lambda _: __import__("socket").gethostname())
                       .collect()))
    print("=" * 70)
    print(f"Spark version        : {spark.version}")
    print(f"Master               : {sc.master}")
    print(f"Default parallelism  : {sc.defaultParallelism}")
    print(f"Hosts in cluster     : {len(hosts)} -> {hosts}")
    print(f"Training partitions  : {parts}")
    print("=" * 70)

    train = load_dataset(spark, args.train).repartition(parts).cache()
    val = load_dataset(spark, args.validation).cache()
    print(f"Training rows: {train.count()}   Validation rows: {val.count()}")

    ev_rmse = RegressionEvaluator(labelCol=LABEL, predictionCol="prediction", metricName="rmse")
    ev_mae = RegressionEvaluator(labelCol=LABEL, predictionCol="prediction", metricName="mae")
    ev_r2 = RegressionEvaluator(labelCol=LABEL, predictionCol="prediction", metricName="r2")

    results, best = [], None
    for name, est in candidates(args.quick):
        t0 = time.time()
        model = Pipeline(stages=feature_stages() + [est]).fit(train)
        tr_pred, va_pred = model.transform(train), model.transform(val)
        r = {
            "model": name,
            "train_rmse": ev_rmse.evaluate(tr_pred),
            "val_rmse": ev_rmse.evaluate(va_pred),
            "val_mae": ev_mae.evaluate(va_pred),
            "val_r2": ev_r2.evaluate(va_pred),
            "fit_seconds": round(time.time() - t0, 2),
        }
        results.append(r)
        print(f"{name:<70} train RMSE={r['train_rmse']:10.4f}  val RMSE={r['val_rmse']:10.4f}  "
              f"R2={r['val_r2']:.6f}  ({r['fit_seconds']}s)")
        if best is None or r["val_rmse"] < best[0]["val_rmse"]:
            best = (r, model)

    best_r, best_model = best
    print("=" * 70)
    print(f"BEST MODEL : {best_r['model']}")
    print(f"Validation RMSE = {best_r['val_rmse']:.6f}   MAE = {best_r['val_mae']:.6f}   "
          f"R2 = {best_r['val_r2']:.8f}")
    lr = best_model.stages[-1]
    if hasattr(lr, "coefficients"):
        print(f"Intercept = {lr.intercept:.4f}")
        print(f"Coefficients = {[round(float(x), 6) for x in lr.coefficients.toArray()]}")
    print("=" * 70)

    best_model.write().overwrite().save(args.model_output)
    print(f"Saved best PipelineModel to {args.model_output}")

    if args.metrics_output:
        report = {"best": best_r, "all_results": sorted(results, key=lambda x: x["val_rmse"]),
                  "hosts": hosts, "default_parallelism": sc.defaultParallelism,
                  "spark_version": spark.version}
        (spark.createDataFrame([(json.dumps(report, indent=2),)], ["value"])
         .coalesce(1).write.mode("overwrite").text(args.metrics_output))
        print(f"Saved metrics to {args.metrics_output}")

    spark.stop()


if __name__ == "__main__":
    main()
