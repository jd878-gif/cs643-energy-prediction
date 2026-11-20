# CS 643 – Programming Assignment 2: Energy Consumption Prediction (Spark MLlib on AWS)

Parallel ML training on a 4-node **AWS EMR Spark cluster**, a single-machine **prediction app** that prints the **RMSE**, and a **Docker** image of that app.

| Item | Link |
|---|---|
| GitHub | `https://github.com/jd878-gif/cs643-energy-prediction` |
| Docker Hub | `https://hub.docker.com/r/dj11108/cs643-energy-predict` |

## Results (validation set, 200 rows)

| Model (best per family) | Validation RMSE | R² |
|---|---|---|
| **Linear Regression (regParam=0) ← selected** | **0.0137** | 1.0000 |
| Linear Regression (regParam=0.01) | 0.0166 | 1.0000 |
| GBT (200 iters, depth 3) | 93.89 | 0.989 |
| Random Forest (50 trees, depth 12) | 180.41 | 0.960 |
| Decision Tree (depth 15) | 222.94 | 0.939 |

Energy consumption is (almost exactly) a linear function of the features: about 0.05·SqFt + 10·Occupants + 20·Appliances − 5·Temp, plus a building-type offset (Industrial +500, Residential −500 vs. Commercial) and weekends being 50 lower than weekdays. So un-regularised linear regression beats every tree model. The remaining 0.0137 error comes from the labels and temperatures being rounded to 2 decimals. No model can go meaningfully lower than that.

## Repository layout
```
training/train.py              Parallel training job (EMR), tunes models on the validation set
prediction/predict.py          Prediction app: takes test-file path, prints RMSE
Dockerfile, .dockerignore      Prediction container (Python 3.11 + Java 17 + PySpark 3.5.1 + model)
requirements.txt
conf/log4j2.properties         Quiet Spark logging inside the container
model/                         Trained Spark PipelineModel (output of the EMR job)
data/                          TrainingDataset.csv, ValidationDataset.csv
scripts/emr_train.sh           One-shot CLI: upload to S3, create 4-node EMR cluster, train, download model
scripts/ec2_setup_prediction.sh  Installs Java/PySpark/Docker on the prediction EC2 instance
```

---

## 1. Parallel model training on EMR (4 EC2 instances)

### 1a. Prepare S3
1. AWS Console → **S3** → *Create bucket*, e.g. `cs643-pa2-<ucid>` (region `us-east-1`).
2. Create the folders `code/` and `data/`, and upload:
   - `training/train.py` → `s3://cs643-pa2-jd878/code/train.py`
   - `data/TrainingDataset.csv`, `data/ValidationDataset.csv` → `s3://cs643-pa2-jd878/data/`

### 1b. Create the EMR cluster
AWS Console → **EMR** → *Create cluster*:
- **Release:** `emr-7.1.0` (or any 7.x). **Application bundle:** *Spark* (Hadoop is included).
- **Instance groups:** Primary = 1 × `m5.xlarge`, Core = **3** × `m5.xlarge` → **4 EC2 instances in total**. Task group: remove it.
- **Security configuration and EC2 key pair:** choose your key pair (AWS Academy: `vockey`).
- **IAM roles:** Service role = `EMR_DefaultRole`, EC2 instance profile = `EMR_EC2_DefaultRole`. In AWS Academy these exist already. Otherwise run `aws emr create-default-roles` once.
- **Cluster logs:** `s3://cs643-pa2-jd878/emr-logs/`.
- Click **Create cluster** and wait until the status is **Waiting** (≈10 min).

### 1c. Run the training job (two options)

**Option A – EMR Step (console):** Cluster → *Steps* → *Add step*
- Type: **Spark application**, Deploy mode: **Cluster**
- Spark-submit options: `--num-executors 6 --executor-cores 2`
- Application location: `s3://cs643-pa2-jd878/code/train.py`
- Arguments:
  ```
  --train s3://cs643-pa2-jd878/data/TrainingDataset.csv
  --validation s3://cs643-pa2-jd878/data/ValidationDataset.csv
  --model-output s3://cs643-pa2-jd878/model
  --metrics-output s3://cs643-pa2-jd878/metrics
  --partitions 12
  ```

**Option B – SSH to the primary node:**
```bash
ssh -i labsuser.pem hadoop@<primary-public-DNS>     # open port 22 in the primary node's security group first
aws s3 cp s3://cs643-pa2-jd878/code/train.py .
spark-submit --master yarn --deploy-mode client --num-executors 6 --executor-cores 2 train.py \
  --train s3://cs643-pa2-jd878/data/TrainingDataset.csv \
  --validation s3://cs643-pa2-jd878/data/ValidationDataset.csv \
  --model-output s3://cs643-pa2-jd878/model --metrics-output s3://cs643-pa2-jd878/metrics --partitions 12
```

**Option C – fully scripted** (from a machine with the AWS CLI): `BUCKET=cs643-pa2-jd878 bash scripts/emr_train.sh`

The job prints the hosts that ran its tasks (`Hosts in cluster : 3 -> [...]`), which shows the training is distributed. It then prints the train/validation RMSE of every candidate model and saves the best `PipelineModel` to `s3://cs643-pa2-jd878/model`. To parallelize, the training data is repartitioned across the executors on the 3 core nodes, and each MLlib `fit()` (L-BFGS/normal-equation aggregation, tree-split statistics) runs as distributed Spark stages. **Terminate the cluster afterwards.**

---

## 2. Prediction on a single EC2 instance – without Docker

1. Launch an EC2 instance: **Ubuntu Server 24.04**, `t2.medium` (or larger), key pair `vockey`, security group allowing SSH (22).
   Under *Advanced details → IAM instance profile*, choose `LabInstanceProfile` (Academy) so the instance can read S3.
2. Connect and set up:
   ```bash
   ssh -i labsuser.pem ubuntu@<ec2-public-DNS>
   git clone https://github.com/jd878-gif/cs643-energy-prediction.git
   cd cs643-energy-prediction
   bash scripts/ec2_setup_prediction.sh          # Java 17, PySpark 3.5.1 venv, Docker
   rm -rf model && aws s3 cp --recursive s3://cs643-pa2-jd878/model model   # model trained on EMR
   ```
3. Run:
   ```bash
   source ~/pa2-venv/bin/activate
   python3 prediction/predict.py data/ValidationDataset.csv --model model
   # any test file:  python3 prediction/predict.py /path/to/TestDataset.csv
   ```
   Output ends with:
   ```
   RMSE = 0.013731
   ```

---

## 3. Prediction with Docker

### Build and push (on the EC2 instance, after step 2)
```bash
docker build -t dj11108/cs643-energy-predict:latest .
docker login
docker push dj11108/cs643-energy-predict:latest
```
*(Optional, so the image also runs natively on Apple-Silicon Macs:*
`docker buildx create --use && docker buildx build --platform linux/amd64,linux/arm64 -t dj11108/cs643-energy-predict:latest --push .`*)*

### Run (any machine with Docker)
The test file is on the host, so mount its folder into the container with `-v` and pass the path *inside* the container:
```bash
sudo docker pull dj11108/cs643-energy-predict:latest
sudo docker run --rm -v /path/to/folder:/data dj11108/cs643-energy-predict:latest /data/TestDataset.csv
# e.g. from the folder that holds the file:
sudo docker run --rm -v "$(pwd)":/data dj11108/cs643-energy-predict:latest /data/TestDataset.csv
```
The image contains the trained model, Java 17, Python 3.11, and PySpark 3.5.1. It needs no Spark cluster.

---

## Robustness of the prediction app
- Auto-detects the delimiter (`,` `;` tab `|`) and handles quoted or differently-cased headers.
- Unseen categorical values do not crash the app (`handleInvalid="keep"`).
- If the label column is missing, it still prints predictions (`--output preds.csv` saves them all).
