#!/bin/bash

# ########################################
# The TensorFlow Serving Docker container is already configured to use port 8500 for gRPC and 8501 for REST.
# ########################################


# ########## Constants: ##########
HOST=8500
PORT=8500
CONTAINER_NAME="tfserving_resnet_similarity"
MODEL_NAME="resnet_similarity"
MODEL_FILE_PATH="/home/deesus/Engineering/PythonProjects/product_similarity/models/resnet_similarity/"
TARGET_PATH_IN_CONTAINER="/models/resnet_similarity"

# ########## Spin up TensorFlow Serving Container: ##########
docker run -p $HOST:$PORT --name $CONTAINER_NAME \
--mount type=bind,source=$MODEL_FILE_PATH,target=$TARGET_PATH_IN_CONTAINER \
-e MODEL_NAME=$MODEL_NAME -t tensorflow/serving
