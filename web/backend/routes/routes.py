from flask import Blueprint, request, jsonify
from flask_cors import CORS
from urllib.request import urlopen
from ..model_client import find_similar_images

api = Blueprint('api', __name__)
CORS(api)


@api.route('/file-upload', methods=['PUT'])
def file_upload():
    uploaded_file = request.files['file']
    if uploaded_file != '':
        # We have to convert file storage object to something cv2/numpy can read:
        # see <https://stackoverflow.com/q/47515243> and <https://stackoverflow.com/q/17170752>:
        uploaded_file = uploaded_file.read()
        # Recall that using `.read()` moves the cursor to EOF; hence the reason we don't check for empty bytes
        # until after (safely) setting the value of `uploaded_file`:
        if uploaded_file != b'':
            # TODO: implement error case if client doesn't return successful response:
            similar_images_ids = find_similar_images(uploaded_file, 'localhost:8500')
            return jsonify(file_paths=similar_images_ids)
    else:
        return jsonify(
            message="No file uploaded",
            category="error",
            status=404
        )


# TODO: handle exceptions for route:
@api.route('/find-related', methods=['PUT'])
def find_related_item():
    img_url = request.get_json()['imgPath']
    response = urlopen(img_url)
    # TODO: is there any real performance gains by converting to bytearray first (rather than directly to bytes)?
    file = bytearray(response.read())
    file = bytes(file)
    similar_images_ids = find_similar_images(file, 'localhost:8500')

    return jsonify(file_paths=similar_images_ids)
