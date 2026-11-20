#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Creates a 4-node EMR Spark cluster (1 primary + 3 core), runs the parallel
# training job as an EMR step, and copies the model back from S3.
# Run from your laptop/AWS CloudShell with the AWS CLI configured
# (AWS Academy: paste the credentials from "AWS Details" into ~/.aws/credentials).
# ----------------------------------------------------------------------------
set -euo pipefail

BUCKET="${BUCKET:?Set BUCKET, e.g. export BUCKET=cs643-pa2-jd878}"
REGION="${REGION:-us-east-1}"
KEY_NAME="${KEY_NAME:-vockey}"            # EC2 key pair name
INSTANCE_TYPE="${INSTANCE_TYPE:-m5.xlarge}"
EMR_RELEASE="${EMR_RELEASE:-emr-7.1.0}"   # Spark 3.5.x

echo ">> Uploading code and data to s3://$BUCKET"
aws s3 mb "s3://$BUCKET" --region "$REGION" 2>/dev/null || true
aws s3 cp training/train.py           "s3://$BUCKET/code/train.py"
aws s3 cp data/TrainingDataset.csv    "s3://$BUCKET/data/TrainingDataset.csv"
aws s3 cp data/ValidationDataset.csv  "s3://$BUCKET/data/ValidationDataset.csv"

echo ">> Creating EMR cluster with the training step"
CLUSTER_ID=$(aws emr create-cluster \
  --region "$REGION" \
  --name "CS643-PA2-Training" \
  --release-label "$EMR_RELEASE" \
  --applications Name=Spark Name=Hadoop \
  --instance-groups \
      InstanceGroupType=MASTER,InstanceCount=1,InstanceType="$INSTANCE_TYPE" \
      InstanceGroupType=CORE,InstanceCount=3,InstanceType="$INSTANCE_TYPE" \
  --ec2-attributes KeyName="$KEY_NAME" \
  --use-default-roles \
  --log-uri "s3://$BUCKET/emr-logs/" \
  --steps "Type=Spark,Name=TrainEnergyModel,ActionOnFailure=CONTINUE,Args=[--deploy-mode,cluster,--num-executors,6,--executor-cores,2,s3://$BUCKET/code/train.py,--train,s3://$BUCKET/data/TrainingDataset.csv,--validation,s3://$BUCKET/data/ValidationDataset.csv,--model-output,s3://$BUCKET/model,--metrics-output,s3://$BUCKET/metrics,--partitions,12]" \
  --auto-terminate \
  --query ClusterId --output text)

echo ">> Cluster $CLUSTER_ID started. Waiting for it to finish (10-15 min)..."
aws emr wait cluster-terminated --cluster-id "$CLUSTER_ID" --region "$REGION"
aws emr list-steps --cluster-id "$CLUSTER_ID" --region "$REGION" \
  --query 'Steps[].{Name:Name,State:Status.State}' --output table

echo ">> Downloading model and metrics"
rm -rf model && aws s3 cp --recursive "s3://$BUCKET/model" model
aws s3 cp --recursive "s3://$BUCKET/metrics" metrics
cat metrics/part-*.txt | head -40
echo ">> Driver log (stdout) is under s3://$BUCKET/emr-logs/$CLUSTER_ID/containers/"
