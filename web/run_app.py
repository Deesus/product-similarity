from flask import request, jsonify
from flask.cli import FlaskGroup
from backend import app
from backend import model_client

cli = FlaskGroup(app)


@app.route('/file-upload', methods=['POST'])
def index():
    uploaded_file = request.files['file']
    if uploaded_file != '':
        # We have to convert file storage object to something cv2/numpy can read:
        # see <https://stackoverflow.com/q/47515243> and <https://stackoverflow.com/q/17170752>:
        uploaded_file = uploaded_file.read()
        # Recall that using `.read()` moves the cursor to EOF; hence the reason we don't check for empty bytes
        # until after (safely) setting the value of `uploaded_file`:
        if uploaded_file != b'':
            # TODO: implement error case if client doesn't return successful response:
            similar_images_ids = model_client.find_similar_images(uploaded_file, 'localhost:8500')
            return jsonify(ids=similar_images_ids)
    else:
        return jsonify(
            message="No file uploaded",
            category="error",
            status=404
        )


# ########## Main: ##########
if __name__ == '__main__':
    cli()
