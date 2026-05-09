# start_mlflow.sh
# Starts the MLflow tracking server in Docker.

######## HOW TO USE ########
# 0. First time only: Run chmod +x start_mlflow.sh
#   (change mode setting executable permission on this script)
# 1. Start your local docker daemon
# 2. Run this script with ./start_mlflow.sh
# 3. Open http://localhost:5000

######## NOTE ########
# The mlflow-data/ volume persists all experiment runs between container restarts.

set -euo pipefail

MLFLOW_PORT=5000
VOLUME_PATH="$(pwd)/mlflow-data"

mkdir -p "$VOLUME_PATH"

# Check if container already running
if docker ps --format '{{.Names}}' | grep -q '^mlflow-server$'; then
    echo "MLflow already running → http://localhost:${MLFLOW_PORT}"
    exit 0
fi

# If container exists but stopped, restart it
if docker ps -a --format '{{.Names}}' | grep -q '^mlflow-server$'; then
    echo "Restarting stopped mlflow-server container..."
    docker start mlflow-server
else
    echo "Starting fresh mlflow-server container..."
    docker run -d \
        --name mlflow-server \
        -p "${MLFLOW_PORT}:5000" \
        -v "${VOLUME_PATH}:/mlflow" \
        ghcr.io/mlflow/mlflow \
        mlflow server --host 0.0.0.0 --backend-store-uri /mlflow
fi

echo "MLflow UI → http://localhost:${MLFLOW_PORT}"