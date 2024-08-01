#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$SCRIPT_DIR

# Set the ini file path 
export OASIS_INI_PATH="${SCRIPT_DIR}/conf.ini"

# Define the path to the version CSV file
VERSION_FILE="${OASIS_MODEL_DATA_DIRECTORY}/keys_data/OasisRed/ModelVersion.csv"

# Check if the version file exists and extract the third value
if [[ -f "$VERSION_FILE" ]]; then
  OASIS_MODEL_VERSION_ID=$(cut -d',' -f3 "$VERSION_FILE")
  export OASIS_MODEL_VERSION_ID
else
  echo "Version file not found at $VERSION_FILE!"
  OASIS_MODEL_VERSION_ID="X.Y.Z"
  export OASIS_MODEL_VERSION_ID
fi

# Delete celeryd.pid file - fix que pickup issues on reboot of server
rm -f /home/worker/celeryd.pid

./src/utils/wait-for-it.sh "$OASIS_RABBIT_HOST:$OASIS_RABBIT_PORT" -t 60
./src/utils/wait-for-it.sh "$OASIS_CELERY_DB_HOST:$OASIS_CELERY_DB_PORT" -t 60

# Start worker on init
celery --app src.model_execution_worker.tasks worker --concurrency=1 --loglevel=INFO -Q "${OASIS_MODEL_SUPPLIER_ID}-${OASIS_MODEL_ID}-${OASIS_MODEL_VERSION_ID}" |& tee -a /var/log/oasis/worker.log
