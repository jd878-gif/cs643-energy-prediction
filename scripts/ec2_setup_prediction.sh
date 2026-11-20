#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Run ON the single Ubuntu EC2 instance used for prediction.
#   1) installs Java 17 + PySpark (prediction without Docker)
#   2) installs Docker (prediction with Docker)
# Usage:  bash scripts/ec2_setup_prediction.sh
# ----------------------------------------------------------------------------
set -euo pipefail

sudo apt-get update -y
sudo apt-get install -y openjdk-17-jre-headless python3-venv python3-pip unzip docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER" || true

# AWS CLI v2 (to pull the model from S3)
if ! command -v aws >/dev/null; then
  curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp && sudo /tmp/aws/install
fi

python3 -m venv ~/pa2-venv
~/pa2-venv/bin/pip install --upgrade pip
~/pa2-venv/bin/pip install -r requirements.txt

echo "Done. Activate with:  source ~/pa2-venv/bin/activate"
echo "Log out and back in once so 'docker' works without sudo."
