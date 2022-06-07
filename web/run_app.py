from flask import request, render_template
from flask.cli import FlaskGroup
from backend import app
from backend import model_client

cli = FlaskGroup(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file != '':
            # We have to convert file storage object to something cv2/numpy can read:
            # see <https://stackoverflow.com/q/47515243> and <https://stackoverflow.com/q/17170752>:
            uploaded_file = uploaded_file.read()
            # Recall that using `.read()` moves the cursor to EOF; hence the reason we don't check for empty bytes
            # until after (safely) setting the value of `uploaded_file`:
            if uploaded_file == b'':
                return render_template('index.html')
            similar_images_ids = model_client.find_similar_images(uploaded_file, 'localhost:8500')
            return render_template('results.html', results=similar_images_ids)
    return render_template('index.html')


# ########## Main: ##########
if __name__ == '__main__':
    cli()
