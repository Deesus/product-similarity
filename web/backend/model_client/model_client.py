from flask import Blueprint
import numpy as np
import cv2
from annoy import AnnoyIndex
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow import make_tensor_proto, make_ndarray
from pathlib import Path

from ..db import get_db_connection

model_client = Blueprint('model_client', __name__)


# ########## Constants: ##########
MODEL_NAME = 'resnet_similarity'  # This cannot be arbitrary; it needs to be the exact name of the saved model.
IMG_HEIGHT = 224
IMG_WIDTH = 224
VECTOR_DIM = 2048  # Equal to final layer (global_max_pooling2d) in model


# ########## Load Annoy index (k-approximate nearest neighbors): ##########
annoy_ = AnnoyIndex(VECTOR_DIM, 'angular')
# File path assumes cwd is the `web/` folder:
annoy_index_path = str(Path('./backend/model_client/annoy_index/abo_embedding.ann').resolve())
annoy_.load(annoy_index_path)


# ########## Helper methods: ##########
def process_image(img: np.ndarray):
    """ Pre-process images before feeding to model.

    Resizes image, scales (/255), and expands array dimension. The model requires specific input dimensions (shape),
    therefore resizing and adding dimension is necessary. Scaling improves performance.
    TODO: ideally, we should use standard deviation and mean of dataset instead of simply dividing by 255.

    :param img:
    :return {np.ndarray}: returns processed image
    """

    # The model signature expects float type as input (enter `saved_model_cli show` in command line for details), but
    # since cv2 reads images as float64, we need to convert it to float32. While this isn't an issue during
    # development/training, when serving via TensorFlow Serving, we would get this error:
    # `Expects arg[0] to be float but double is provided`. N.b. np.float64 and np.float32 correspond to double and float
    # types, respectively (in Java, C++, etc.); hence the error message, and reason for casting to float32:
    processed_img = np.float32(img)

    # see comparison of different interpolation methods: <https://chadrick-kwag.net/cv2-resize-interpolation-methods/>
    processed_img = cv2.resize(processed_img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)

    # We can use TF's ResNet `preprocess_input`:
    # <https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2/preprocess_input>; however, because
    # the datasets are quite different, it might do more harm than good <https://stackoverflow.com/a/58250681>; we can
    # therefore, simply divide by 255:
    processed_img = processed_img / 255

    # ResNet model expects input shape (batch_size, img_height, img_width, channels); therefore, we need to add
    # batch_size dimension:
    processed_img = np.expand_dims(processed_img, axis=0)
    return processed_img


def query_product_by_id(id: int, cursor):
    return cursor.execute('SELECT * FROM products WHERE id=?', (id,)).fetchone()


# TODO: need to filter uploaded files to only accept images:
def find_similar_images(file: bytes, server_address: str, num_results: int = 6):
    # We have to convert byte string for cv2/numpy; see <https://stackoverflow.com/q/17170752>:
    img_np = np.fromstring(file, np.uint8)  # (image) file represented as numpy array
    img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)

    processed_image = process_image(img)

    # Get embedding from model server:
    target_embedding = get_prediction(processed_image, server_address)

    # Let Annoy find top matches:
    ids_of_matches = annoy_.get_nns_by_vector(target_embedding, num_results)

    # Get list of image paths from db:
    conn = get_db_connection()
    cursor = conn.cursor()
    list_of_file_paths = [query_product_by_id(id, cursor)['file_path'] for id in ids_of_matches]
    conn.close()

    return list_of_file_paths


def get_prediction(img: np.ndarray, server_address: str):
    """

    See <https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-tfserving.html> and
    <https://www.tensorflow.org/tfx/serving/docker> as general guides.

    Both the `inputs` "name" and `model_spec.signature_name` are required for the `request`, and they both can
    be discerned by entering the following in the terminal:
    `saved_model_cli show --dir PATH/TO/MODEL/MODEL_VERSION --all`
    After doing so, you'll see a printout similar to this:
    ```
    signature_def['serving_default']:
      The given SavedModel SignatureDef contains the following input(s):
        inputs['resnet50v2_input'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1, 224, 224, 3)
            name: serving_default_resnet50v2_input:0
      The given SavedModel SignatureDef contains the following output(s):
        outputs['global_max_pooling2d_5'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1, 2048)
            name: StatefulPartitionedCall:0
      Method name is: tensorflow/serving/predict
    ```
    So, in this example, we'd want to use `signature_def['serving_default']` for the `signature_name`, and
    `inputs['resnet50v2_input']` when we call `CopyFrom()`.
    :return:
    """
    channel = grpc.insecure_channel(server_address)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # See above docstring on how `input_name` and `output_name` values are set:
    input_name = 'resnet50v2_input'
    output_name = 'global_max_pooling2d'
    timeout = 6.0  # in seconds

    request = predict_pb2.PredictRequest()
    request.model_spec.name = MODEL_NAME
    # See above docstring on how `signature_name` value is set:
    request.model_spec.signature_name = 'serving_default'

    request.inputs[input_name].CopyFrom(
        make_tensor_proto(img, shape=(1, IMG_HEIGHT, IMG_WIDTH, 3))
    )
    response = stub.Predict(request, timeout)

    # Convert TensorProto values into numpy array:
    # `output` embedding is rank-1 array with shape = (VECTOR_DIM,):
    output = make_ndarray(response.outputs[output_name])[-1]
    return output
